"""Training loop for the Wordle GPT.

The dataset is a single token stream (see data.py). Each step we sample
`batch_size` random blocks of `block_size` tokens, shift by one, and
minimize cross-entropy only at positions whose target is a word token.

Evaluation during training plays `eval_games` fresh games with greedy
decoding and reports solve rate + average guesses, so you watch the model
actually learn to play, not just the loss drop.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from .model import GPT, GPTConfig


def _random_blocks(tokens: np.ndarray, n: int, block_size: int, device: str, rng: np.random.Generator):
    hi = len(tokens) - block_size - 1
    starts = rng.integers(0, hi, size=n)
    idxs = np.stack([tokens[s : s + block_size] for s in starts]).astype(np.int64)
    idx = torch.from_numpy(idxs).to(device)
    targets = torch.from_numpy(
        np.stack([tokens[s + 1 : s + block_size + 1] for s in starts]).astype(np.int64)
    ).to(device)
    return idx, targets


def word_mask_targets(targets: torch.Tensor, word_offset: int, word_count: int) -> torch.Tensor:
    """Mask targets to -1 (ignored) unless they are word tokens.

    Only word positions carry a training signal: the model should produce a
    guess there and nothing anywhere else.
    """
    m = (targets >= word_offset) & (targets < word_offset + word_count)
    return torch.where(m, targets, torch.full_like(targets, -1))


def train(
    tokens: np.ndarray,
    cfg: GPTConfig,
    device: str = "cpu",
    out_dir: str | Path = "checkpoints",
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = 3e-4,
    warmup_steps: int = 200,
    weight_decay: float = 0.05,
    grad_clip: float = 1.0,
    log_every: int = 50,
    eval_every: int = 250,
    eval_games: int = 100,
    eval_fn=None,
    seed: int = 0,
    run_name: str = "wordle",
    word_count: int | None = None,
):
    """Train `cfg`-shaped GPT on the token stream.

    eval_fn(step, model) -> dict is called every eval_every steps (e.g. to
    report live solve rate); if None, only loss is reported.
    Returns (model, history) where history is a list of per-log rows.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params on {device}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    model.train()
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        idx, targets = _random_blocks(tokens, batch_size, cfg.block_size, device, rng)
        if word_count is None:
            word_count = cfg.vocab_size  # legacy default keeps behavior sane
        targets = word_mask_targets(targets, 2, word_count)
        logits, loss = model(idx, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            el = time.time() - t0
            rate = step * batch_size * cfg.block_size / el
            print(f"step {step:5d} loss {loss.item():.4f} lr {lr_now:.2e} tok/s {rate:,.0f}")
            history.append({"step": step, "loss": loss.item(), "lr": lr_now})

        if eval_every and step % eval_every == 0:
            extra = {}
            if eval_fn is not None:
                extra = eval_fn(step, model)
            if extra:
                print(f"  eval step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in extra.items()))
                history.append({"step": step, **extra})

        if step % max(1, steps // 5) == 0 or step == steps:
            model.save(out_dir / f"{run_name}-step{step}.pt")

    model.save(out_dir / f"{run_name}-final.pt")
    print(f"done in {time.time() - t0:.0f}s; checkpoint -> {out_dir}/{run_name}-final.pt")
    return model, history

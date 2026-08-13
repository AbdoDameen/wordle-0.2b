"""Evaluation: let the model (or the baseline) actually play Wordle.

play_games runs N games with batched inference: all unfinished games share
one forward pass per move, so evaluating 500 games is seconds on GPU and
cheap on CPU.
"""
from __future__ import annotations

import numpy as np
import torch

from .baseline import EntropySolver
from .game import WordleGame
from .words import Vocabulary


def split_answers(vocab: Vocabulary, holdout_size: int = 200, seed: int = 42) -> tuple[list[int], list[int]]:
    """Split answers into (train, holdout) word ids.

    Use `train` as the secret pool for data generation and `holdout` for
    evaluation. The holdout answers are never seen as secrets during
    training, so the model's solve rate on them is honest generalization,
    not memorization of a trajectory.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(vocab.answers))
    holdout = idx[:holdout_size]
    train = idx[holdout_size:]
    to_wid = np.asarray(vocab.answer_ids, dtype=np.int64)
    return to_wid[train].tolist(), to_wid[holdout].tolist()


@torch.no_grad()
def next_guesses(
    model, vocab: Vocabulary, games: list[WordleGame], device: str, temperature: float = 0.0,
    consistent: bool = True,
) -> list[str]:
    """One batched forward pass; returns a guess per unfinished game.

    With consistent=True (default) any guess that contradicts the feedback
    seen so far is masked out. Such a guess can never be the secret, so this
    only removes objectively bad moves.
    """
    if not games:
        return []
    max_t = max(len(g.guesses) for g in games)
    batch = []
    for g in games:
        prefix = vocab.encode_prefix(g.guesses, g.feedbacks)
        batch.append(prefix + [vocab.PAD] * (max_t - len(g.guesses)))
    idx = torch.tensor(batch, dtype=torch.long, device=device)
    logits = model(idx)[0][:, -1, :]  # (B, V) at last position
    # mask to word tokens only
    word_mask = torch.zeros(vocab.VOCAB_SIZE, dtype=torch.bool, device=device)
    word_mask[vocab.WORD_OFFSET : vocab.WORD_OFFSET + vocab.V] = True
    logits = torch.where(word_mask, logits, torch.full_like(logits, float("-inf")))
    if consistent:
        masks = consistent_word_masks(vocab, games)
        logits = logits.masked_fill(~torch.from_numpy(masks).to(device), float("-inf"))
    for i, g in enumerate(games):
        for w in g.guesses:
            logits[i, vocab.word_id(w)] = float("-inf")  # no repeats
    if temperature <= 0:
        ids = logits.argmax(dim=-1)
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        ids = torch.multinomial(probs, 1).squeeze(-1)
    return [vocab.word(int(i)) for i in ids]


_WORDS_BYTES_CACHE = {}


def _words_bytes(vocab: Vocabulary) -> np.ndarray:
    if id(vocab) not in _WORDS_BYTES_CACHE:
        _WORDS_BYTES_CACHE[id(vocab)] = np.array(
            [list(w.encode()) for w in vocab.words], dtype=np.uint8
        ) - ord("a")
    return _WORDS_BYTES_CACHE[id(vocab)]


def consistent_word_masks(vocab: Vocabulary, games: list[WordleGame]) -> np.ndarray:
    """(len(games), V) bool: which words are consistent with each game's feedback."""
    from .baseline import pattern_ids_for_guess

    W = _words_bytes(vocab)
    masks = np.ones((len(games), vocab.V), dtype=bool)
    for gi, g in enumerate(games):
        for guess, fb in zip(g.guesses, g.feedbacks):
            pat = pattern_ids_for_guess(W, W[vocab.word_id(guess)])
            fb_pid = sum(3**i * {"G": 2, "Y": 1, "B": 0}[c] for i, c in enumerate(reversed(fb)))
            masks[gi] &= pat == fb_pid
    return masks


@torch.no_grad()
def play_games(
    model,
    vocab: Vocabulary,
    n_games: int,
    secrets: list[str],
    device: str = "cpu",
    temperature: float = 0.0,
    verbose: bool = False,
) -> list[WordleGame]:
    """Play n_games with the model; `secrets` must have at least n_games words."""
    model.eval()
    games = [WordleGame(secrets[i]) for i in range(n_games)]
    finished = [False] * n_games
    for _ in range(6):
        active = [g for g, f in zip(games, finished) if not f]
        if not active:
            break
        guesses = next_guesses(model, vocab, active, device, temperature)
        it = iter(guesses)
        for g, f in zip(games, finished):
            if f:
                continue
            guess = next(it)
            g.play(guess)
            if g.solved:
                finished[games.index(g)] = True
    return games


def summarize(games: list[WordleGame]) -> dict:
    solved = [g for g in games if g.solved]
    dist = {k: sum(1 for g in solved if len(g.guesses) == k) for k in range(1, 7)}
    return {
        "games": len(games),
        "solve_rate": len(solved) / len(games),
        "avg_guesses": float(np.mean([len(g.guesses) for g in solved])) if solved else float("nan"),
        "median_guesses": float(np.median([len(g.guesses) for g in solved])) if solved else float("nan"),
        "guess_distribution": dist,
        "lost": sum(1 for g in games if not g.solved),
    }


def play_games_baseline(
    vocab: Vocabulary, pattern_matrix: np.ndarray, n_games: int, secrets: list[str], seed: int = 0
) -> list[WordleGame]:
    solver = EntropySolver(vocab, pattern_matrix)
    rng = np.random.default_rng(seed)
    games = []
    for s in secrets[:n_games]:
        games.append(solver.play(s, rng=rng))
    return games

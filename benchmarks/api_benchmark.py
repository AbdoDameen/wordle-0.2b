#!/usr/bin/env python3
"""Benchmark how different LLMs (and local solvers) play Wordle.

Every contestant plays the same N games against the same secret words and
the same rules. Chat models get the game history as a conversation and must
reply with a single 5-letter word; the local model and the entropy baseline
play through the same game engine.

Providers (OpenAI-compatible chat completions, unless noted):
    openai/...      OPENAI_API_KEY
    anthropic/...   ANTHROPIC_API_KEY   (native Messages API)
    deepseek/...    DEEPSEEK_API_KEY
    groq/...        GROQ_API_KEY
    openrouter/...  OPENROUTER_API_KEY
    together/...    TOGETHER_API_KEY
    gemini/...      GEMINI_API_KEY      (OpenAI-compat endpoint)
    local/...       LOCAL_API_BASE + optional LOCAL_API_KEY (vLLM, Ollama, LM Studio)
No API needed:
    baseline                          entropy solver (the teacher)
    local0.2b                         the trained checkpoint (--checkpoint)

Example:
    OPENAI_API_KEY=sk-... DEEPSEEK_API_KEY=sk-... python benchmarks/api_benchmark.py \
        --models openai/gpt-4o-mini,deepseek/deepseek-chat,baseline \
        --games 30 --out results

Costs are approximate and read from a small pricing table; unknown models
report cost as None.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import requests

# --------------------------------------------------------------------------
# providers & pricing
# --------------------------------------------------------------------------

OPENAI_COMPAT = {
    "openai": {"base": "https://api.openai.com/v1", "env": "OPENAI_API_KEY"},
    "deepseek": {"base": "https://api.deepseek.com/v1", "env": "DEEPSEEK_API_KEY"},
    "groq": {"base": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY"},
    "openrouter": {"base": "https://openrouter.ai/api/v1", "env": "OPENROUTER_API_KEY"},
    "together": {"base": "https://api.together.xyz/v1", "env": "TOGETHER_API_KEY"},
    "gemini": {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "env": "GEMINI_API_KEY"},
    "local": {"base": None, "env": "LOCAL_API_KEY", "base_env": "LOCAL_API_BASE"},
}

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

# approximate USD per 1M tokens, input/output. Substring-matched against the
# model name. Prices drift; treat as estimates.
PRICING = [
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4.1-mini", 0.40, 1.60),
    ("gpt-4.1", 2.00, 8.00),
    ("gpt-4o", 2.50, 10.00),
    ("o4-mini", 1.10, 4.40),
    ("o3-mini", 1.10, 4.40),
    ("claude-sonnet-4-5", 3.00, 15.00),
    ("claude-opus-4-5", 5.00, 25.00),
    ("claude-3-7-sonnet", 3.00, 15.00),
    ("claude-3-5-haiku", 0.80, 4.00),
    ("deepseek-reasoner", 0.55, 2.19),
    ("deepseek-chat", 0.27, 1.10),
    ("llama-3.3-70b", 0.59, 0.79),
    ("llama-4-maverick", 0.20, 0.60),
    ("gemini-2.5-pro", 1.25, 10.00),
    ("gemini-2.5-flash", 0.30, 2.50),
]

SYSTEM_PROMPT = (
    "You are playing Wordle. The secret is a 5-letter English word. "
    "After each guess you receive 5 letters of feedback: "
    "G = correct letter in the correct position, Y = letter is in the word but in the wrong position, "
    "B = letter is not in the word at all. "
    "Reply with exactly one valid 5-letter English word per message. Nothing else. You have 6 guesses."
)


def price_for(model: str) -> tuple[float, float] | None:
    for key, p_in, p_out in PRICING:
        if key in model:
            return p_in, p_out
    return None


def parse_guess(text: str) -> str | None:
    """Extract a 5-letter word from a model reply."""
    t = text.strip().strip("`\"").strip()
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = t.replace("**", "").replace('"', " ").replace("'", " ").replace(".", " ")
    for tok in re.findall(r"[a-zA-Z]+", t.lower()):
        if len(tok) == 5:
            return tok
    return None


# --------------------------------------------------------------------------
# clients
# --------------------------------------------------------------------------

def client_openai_compat(provider: str, model: str):
    cfg = OPENAI_COMPAT[provider]
    base = cfg["base"]
    if base is None:  # local provider
        base = __import__("os").environ.get("LOCAL_API_BASE")
        if not base:
            raise RuntimeError("LOCAL_API_BASE not set for the local provider")
    api_key = __import__("os").environ.get(cfg["env"], "")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def call(messages, temperature=0.7):
        body = {"model": model, "messages": messages, "max_tokens": 24, "temperature": temperature}
        t0 = time.time()
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        j = r.json()
        return {
            "text": j["choices"][0]["message"]["content"],
            "usage": j.get("usage", {}),
            "latency": time.time() - t0,
        }

    return call


def client_anthropic(model: str):
    api_key = __import__("os").environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    url = f"{ANTHROPIC_BASE}/messages"
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}

    def call(messages, temperature=0.7):
        system = messages[0]["content"]
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages[1:]]
        body = {"model": model, "system": system, "messages": msgs, "max_tokens": 24, "temperature": temperature}
        t0 = time.time()
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        j = r.json()
        return {
            "text": j["content"][0]["text"],
            "usage": {"prompt_tokens": j["usage"]["input_tokens"], "completion_tokens": j["usage"]["output_tokens"]},
            "latency": time.time() - t0,
        }

    return call


def make_client(provider: str, model: str):
    if provider == "anthropic":
        return client_anthropic(model)
    return client_openai_compat(provider, model)


# --------------------------------------------------------------------------
# game protocol
# --------------------------------------------------------------------------

def play_api_game(call, model: str, vocab, secret: str, max_guesses: int = 6) -> dict:
    """One game against the API model. Returns the game record."""
    from wordle02b.game import feedback  # local import keeps CLI light

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    guesses, feedbacks = [], []
    usage_in, usage_out = 0, 0
    latency = 0.0
    retries = 0
    error = None

    for _ in range(max_guesses):
        attempts = 0
        guess = None
        while guess is None and attempts < 2:
            attempts += 1
            try:
                out = call(messages, temperature=0.7)
            except Exception as e:  # network / 5xx / timeout
                error = f"api_error: {type(e).__name__}: {e}"
                time.sleep(1.5)
                if attempts == 2:
                    return _record(guesses, feedbacks, usage_in, usage_out, latency, retries, error, "error")
                continue
            latency += out["latency"]
            usage_in += out["usage"].get("prompt_tokens", 0)
            usage_out += out["usage"].get("completion_tokens", 0)
            w = parse_guess(out["text"])
            if w is None or w not in vocab.word_to_id:
                retries += 1
                messages.append({"role": "user", "content": "That reply was not a single valid 5-letter word. Reply with exactly one valid 5-letter English word."})
                continue
            if w in guesses:
                retries += 1
                messages.append({"role": "user", "content": f"You already guessed {w.upper()}. Pick a different word."})
                continue
            guess = w

        if guess is None:
            return _record(guesses, feedbacks, usage_in, usage_out, latency, retries, "invalid_response", "invalid")

        fb = feedback(guess, secret)
        guesses.append(guess)
        feedbacks.append(fb)
        if guess == secret:
            return _record(guesses, feedbacks, usage_in, usage_out, latency, retries, None, "solved")
        messages.append({"role": "assistant", "content": guess})
        messages.append({"role": "user", "content": fb})

    return _record(guesses, feedbacks, usage_in, usage_out, latency, retries, None, "lost")


def _record(guesses, feedbacks, usage_in, usage_out, latency, retries, error, outcome):
    return {
        "outcome": outcome,
        "guesses_used": len(guesses),
        "solved": outcome == "solved",
        "retries": retries,
        "latency_s": round(latency, 2),
        "tokens_in": usage_in,
        "tokens_out": usage_out,
        "error": error,
        "guess_history": list(zip(guesses, feedbacks)),
    }


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def run_benchmark(
    models: list[tuple[str, str]],
    secrets: list[str],
    vocab,
    checkpoint: str | None = None,
    verbose: bool = True,
) -> dict:
    """models = [(provider, model_name)]. Returns {entry: rows}."""
    rows: dict[str, list[dict]] = {}
    for provider, model in models:
        tag = model if provider in ("baseline", "local0.2b") else f"{provider}/{model}"
        print(f"\n=== {tag} ===")
        if provider == "baseline":
            from wordle02b.baseline import EntropySolver, build_pattern_matrix_cached
            from wordle02b.evaluate import summarize

            P = build_pattern_matrix_cached(vocab, cache_dir="data/cache")
            solver = EntropySolver(vocab, P)
            games = [solver.play(s) for s in secrets]
            st = summarize(games)
            print(f"  solve_rate={st['solve_rate']:.3f} avg={st['avg_guesses']:.2f}")
            rows[tag] = [
                {
                    "outcome": "solved" if g.solved else "lost",
                    "guesses_used": len(g.guesses),
                    "solved": g.solved,
                    "retries": 0,
                    "latency_s": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": None,
                    "guess_history": list(zip(g.guesses, g.feedbacks)),
                }
                for g in games
            ]
            continue

        if provider == "local0.2b":
            import torch

            from wordle02b.evaluate import play_games, summarize
            from wordle02b.model import GPT

            if not checkpoint:
                print("  skipped: pass --checkpoint to benchmark the local model")
                rows[tag] = []
                continue
            model_obj = GPT.load(checkpoint, device="cpu")
            games = play_games(model_obj, vocab, len(secrets), secrets, device="cpu")
            st = summarize(games)
            print(f"  solve_rate={st['solve_rate']:.3f} avg={st['avg_guesses']:.2f}")
            rows[tag] = [
                {
                    "outcome": "solved" if g.solved else "lost",
                    "guesses_used": len(g.guesses),
                    "solved": g.solved,
                    "retries": 0,
                    "latency_s": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": None,
                    "guess_history": list(zip(g.guesses, g.feedbacks)),
                }
                for g in games
            ]
            continue

        call = make_client(provider, model)
        price = price_for(model)
        recs = []
        for i, secret in enumerate(secrets):
            rec = play_api_game(call, model, vocab, secret)
            rec["secret"] = secret
            rec["price"] = price
            if price:
                rec["cost_usd"] = (rec["tokens_in"] * price[0] + rec["tokens_out"] * price[1]) / 1e6
            else:
                rec["cost_usd"] = None
            recs.append(rec)
            if verbose:
                mark = {1: "in 1", 2: "in 2", 3: "in 3", 4: "in 4", 5: "in 5", 6: "in 6"}.get(
                    rec["guesses_used"] if rec["solved"] else None, "lost"
                )
                print(f"  game {i+1:2d}/{len(secrets)}: {secret} -> {mark}")
        rows[tag] = recs
        solved = sum(r["solved"] for r in recs)
        avg = np.mean([r["guesses_used"] for r in recs if r["solved"]]) if solved else float("nan")
        print(f"  solve_rate={solved/len(recs):.3f} avg_guesses={avg:.2f} cost={sum(r['cost_usd'] or 0 for r in recs):.4f}")
    return rows


def summarize_rows(rows: dict[str, list[dict]]) -> list[dict]:
    out = []
    for tag, recs in rows:
        if not recs:
            continue
        n = len(recs)
        solved = [r for r in recs if r["solved"]]
        out.append(
            {
                "model": tag,
                "games": n,
                "solve_rate": len(solved) / n,
                "avg_guesses": round(float(np.mean([r["guesses_used"] for r in solved])), 2) if solved else None,
                "lost": n - len(solved),
                "invalid": sum(1 for r in recs if r["outcome"] == "invalid"),
                "api_errors": sum(1 for r in recs if r["outcome"] == "error"),
                "avg_latency_s": round(float(np.mean([r["latency_s"] for r in recs])), 2),
                "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in recs), 4),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="Benchmark Wordle solving across LLM APIs and local solvers")
    ap.add_argument("--models", required=True, help="comma list: openai/gpt-4o-mini,deepseek/deepseek-chat,baseline,...")
    ap.add_argument("--games", type=int, default=30, help="games per contestant (default 30)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default=None, help="checkpoint for the local0.2b entry")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--plot", action="store_true", help="render a comparison chart")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from wordle02b import Vocabulary, load_word_lists

    vocab = Vocabulary(*load_word_lists())
    rng = np.random.default_rng(args.seed)
    secrets = [str(rng.choice(vocab.answers)) for _ in range(args.games)]

    entries = [tuple(e.split("/", 1)) if "/" in e else (e, e) for e in args.models.split(",")]
    known = {"baseline", "local0.2b"} | set(OPENAI_COMPAT)
    for prov, _ in entries:
        if prov not in known:
            print(f"unknown provider '{prov}' (known: {', '.join(sorted(known))})")
            sys.exit(1)

    rows = run_benchmark(entries, secrets, vocab, checkpoint=args.checkpoint)
    table = summarize_rows(list(rows.items()))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"wordle_benchmark_{stamp}"
    with open(f"{base}.json", "w") as f:
        json.dump({"secrets": secrets, "rows": rows, "table": table}, f, indent=2, default=str)
    with open(f"{base}.csv", "w") as f:
        import csv

        cols = ["model", "games", "solve_rate", "avg_guesses", "lost", "invalid", "api_errors", "avg_latency_s", "total_cost_usd"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow(r)

    print("\n===== results =====")
    for r in sorted(table, key=lambda x: -x["solve_rate"]):
        print(
            f"{r['model']:35s} solve={r['solve_rate']:.3f} avg={r['avg_guesses']} "
            f"lost={r['lost']} invalid={r['invalid']} lat={r['avg_latency_s']}s cost=${r['total_cost_usd']}"
        )
    print(f"saved -> {base}.json / {base}.csv")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
            labels = [r["model"] for r in sorted(table, key=lambda x: -x["solve_rate"])]
            vals = [r["solve_rate"] * 100 for r in sorted(table, key=lambda x: -x["solve_rate"])]
            colors = ["#6aaa64" if v == max(vals) else "#c9b458" if v > 60 else "#787c7e" for v in vals]
            ax.barh(labels, vals, color=colors)
            ax.set_xlabel("solve rate (%)")
            ax.set_title("Wordle solve rate by model")
            ax.set_xlim(0, 105)
            for i, v in enumerate(vals):
                ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(f"{base}.png", bbox_inches="tight")
            print(f"chart -> {base}.png")
        except Exception as e:
            print(f"chart failed: {e}")


if __name__ == "__main__":
    main()

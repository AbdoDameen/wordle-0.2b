"""Wordle-0.2b web UI (Gradio).

Two modes:
  - you play, with an optional AI hint button
  - autoplay: the model (or the entropy baseline if no checkpoint) plays

Run:  python app/gradio_app.py [--checkpoint checkpoints/wordle-0.2b-final.pt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr
import numpy as np

from wordle02b import Vocabulary, load_word_lists
from wordle02b.baseline import EntropySolver, build_pattern_matrix_cached
from wordle02b.evaluate import play_games, summarize
from wordle02b.game import feedback

answers, allowed = load_word_lists()
vocab = Vocabulary(answers, allowed)
P = build_pattern_matrix_cached(vocab, cache_dir="data/cache")
solver = EntropySolver(vocab, P)

try:
    import torch

    _DEVICE = (
        "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
    )
except Exception:
    _DEVICE = "cpu"

TILE = {
    "G": ("#6aaa64", "#ffffff"),
    "Y": ("#c9b458", "#ffffff"),
    "B": ("#787c7e", "#ffffff"),
}
EMPTY = ("#121213", "#787c7e")


def tile_html(letter: str, color: str) -> str:
    bg, fg = TILE[color]
    return f'<div style="width:52px;height:52px;background:{bg};color:{fg};font:bold 28px sans-serif;display:flex;align-items:center;justify-content:center;border-radius:6px;margin:3px">{letter}</div>'


def board_html(guesses: list[str], feedbacks: list[str], max_guesses: int = 6) -> str:
    rows = []
    for i in range(max_guesses):
        if i < len(guesses):
            cells = "".join(tile_html(l.upper(), c) for l, c in zip(guesses[i], feedbacks[i]))
        else:
            cells = "".join(tile_html("", "B") for _ in range(5))
        rows.append(f'<div style="display:flex;justify-content:center">{cells}</div>')
    return f'<div style="font-family:sans-serif;padding:12px;background:#121213;border-radius:10px">{chr(10).join(rows)}</div>'


class State:
    def __init__(self):
        self.secret = ""
        self.guesses: list[str] = []
        self.feedbacks: list[str] = []
        self.new_game()

    def new_game(self):
        self.secret = str(np.random.default_rng().choice(vocab.answers))
        self.guesses, self.feedbacks = [], []

    def apply(self, guess: str) -> tuple[str, bool]:
        guess = guess.strip().lower()
        if len(guess) != 5 or guess not in vocab.word_to_id:
            return "not a valid word", False
        fb = feedback(guess, self.secret)
        self.guesses.append(guess)
        self.feedbacks.append(fb)
        if guess == self.secret:
            return "solved", True
        if len(self.guesses) >= 6:
            return f"out of guesses, word was {self.secret.upper()}", True
        return "", False


state = State()


def render() -> tuple[str, str]:
    return board_html(state.guesses, state.feedbacks), state.secret.upper()


def on_guess(word: str) -> tuple[str, str, str]:
    msg, done = state.apply(word)
    board, _ = render()
    status = msg or f"guess {len(state.guesses)}/6"
    if done:
        status = msg
    return board, status, ""


def on_hint() -> tuple[str, str]:
    from wordle02b.evaluate import next_guesses
    from wordle02b.game import WordleGame

    g = WordleGame(state.secret)
    g.guesses, g.feedbacks = list(state.guesses), list(state.feedbacks)
    hint = next_guesses(model, vocab, [g], device=_DEVICE)[0] if model is not None else "no model loaded"
    board, _ = render()
    return board, f"hint: {hint.upper()}"


def on_autoplay(progress: gr.Progress) -> tuple[str, str, str]:
    state.new_game()
    for _ in range(6):
        from wordle02b.evaluate import next_guesses
        from wordle02b.game import WordleGame

        g = WordleGame(state.secret)
        g.guesses, g.feedbacks = list(state.guesses), list(state.feedbacks)
        guess = next_guesses(model, vocab, [g], device=_DEVICE)[0] if model is not None else solver.play(state.secret).guesses[-1]
        msg, done = state.apply(guess)
        progress(0.2 * (len(state.guesses) - 1))
        if done:
            break
    board, _ = render()
    if state.secret in state.guesses:
        status = f"autoplay solved in {len(state.guesses)}, word was {state.secret.upper()}"
    else:
        status = f"autoplay lost, word was {state.secret.upper()}"
    return board, status, ""


def on_new() -> tuple[str, str, str]:
    state.new_game()
    board, _ = render()
    return board, "new game", ""


def main():
    global model
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    model = None
    if args.checkpoint and Path(args.checkpoint).exists():
        import torch

        from wordle02b.model import GPT

        model = GPT.load(args.checkpoint, device=_DEVICE)
        print(f"loaded {args.checkpoint} on {_DEVICE}")

    with gr.Blocks(theme=gr.themes.Base(primary_hue="green"), title="wordle-0.2b") as demo:
        gr.Markdown("# wordle-0.2b\nA 0.2B transformer that plays Wordle. Play yourself, ask for a hint, or watch it autoplay.")
        board = gr.HTML(value=render()[0])
        status = gr.Markdown("new game")
        with gr.Row():
            guess_box = gr.Textbox(placeholder="5-letter guess", max_lines=1, scale=3)
            guess_btn = gr.Button("Guess", variant="primary")
        with gr.Row():
            hint_btn = gr.Button("Hint")
            auto_btn = gr.Button("Autoplay")
            new_btn = gr.Button("New game")
        guess_btn.click(on_guess, [guess_box], [board, status, guess_box])
        guess_box.submit(on_guess, [guess_box], [board, status, guess_box])
        hint_btn.click(on_hint, [], [board, status])
        auto_btn.click(on_autoplay, [], [board, status, guess_box])
        new_btn.click(on_new, [], [board, status, guess_box])
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

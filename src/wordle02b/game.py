"""Wordle game engine. Feedback letters: G green, Y yellow, B grey."""
from __future__ import annotations

from collections import Counter

WORD_LEN = 5


def feedback(guess: str, secret: str) -> str:
    """Standard Wordle feedback, double-letter rules included."""
    assert len(guess) == WORD_LEN and len(secret) == WORD_LEN
    result = ["B"] * WORD_LEN
    remaining = list(secret)
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            result[i] = "G"
            remaining[i] = None
    counts = Counter(c for c in remaining if c is not None)
    for i, g in enumerate(guess):
        if result[i] == "B" and counts.get(g, 0) > 0:
            result[i] = "Y"
            counts[g] -= 1
    return "".join(result)


class WordleGame:
    """Tracks one game. play() returns the feedback for the guess."""

    def __init__(self, secret: str, max_guesses: int = 6):
        self.secret = secret
        self.max_guesses = max_guesses
        self.guesses: list[str] = []
        self.feedbacks: list[str] = []
        self.solved = False

    def play(self, guess: str) -> str:
        if self.over:
            raise ValueError("game already over")
        fb = feedback(guess, self.secret)
        self.guesses.append(guess)
        self.feedbacks.append(fb)
        if guess == self.secret:
            self.solved = True
        return fb

    @property
    def over(self) -> bool:
        return self.solved or len(self.guesses) >= self.max_guesses

    @property
    def guesses_left(self) -> int:
        return self.max_guesses - len(self.guesses)

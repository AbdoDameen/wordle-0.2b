"""Word lists and vocabulary handling.

The corpus ships with the repo (see data/official/): ~12.9k legal guess
words, ~2.3k of which are possible answers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "official"


@lru_cache(maxsize=1)
def load_word_lists(data_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Return (answers, allowed). Both are sorted lists of lowercase words."""
    d = Path(data_dir) if data_dir else DATA_DIR
    answers = sorted((d / "answers.txt").read_text().split())
    allowed = sorted((d / "allowed.txt").read_text().split())
    return answers, allowed


class Vocabulary:
    """Maps words and feedback letters to integer ids.

    Token layout:
        PAD = 0
        START = 1
        WORD_OFFSET .. WORD_OFFSET + V - 1  -> one token per legal guess word
        G, Y, B                             -> feedback letters
    """

    def __init__(self, answers: list[str], allowed: list[str]):
        # Union keeps answers addressable as words while allowing extra guesses.
        words = sorted(set(answers) | set(allowed))
        missing = set(answers) - set(words)
        if missing:
            raise ValueError(f"answers not in vocab: {sorted(missing)[:5]}...")
        self.words = words
        self.answers = sorted(set(answers))
        self.word_to_id = {w: i for i, w in enumerate(words)}
        self.answer_ids = [self.word_to_id[w] for w in self.answers]
        self.V = len(words)

        self.PAD = 0
        self.START = 1
        self.WORD_OFFSET = 2
        self.G = self.WORD_OFFSET + self.V
        self.Y = self.G + 1
        self.B = self.G + 2
        self.VOCAB_SIZE = self.B + 1

        self.fb_to_token = {"G": self.G, "Y": self.Y, "B": self.B}
        self.token_to_fb = {self.G: "G", self.Y: "Y", self.B: "B"}

    # -- words -----------------------------------------------------------
    def word_id(self, word: str) -> int:
        return self.word_to_id[word]

    def word(self, word_id: int) -> str:
        return self.words[word_id]

    def is_word_token(self, token: int) -> bool:
        return self.WORD_OFFSET <= token < self.WORD_OFFSET + self.V

    def word_mask(self, device="cpu") -> "torch.Tensor":
        """Boolean mask over the vocab selecting word tokens (for sampling)."""
        import torch

        m = torch.zeros(self.VOCAB_SIZE, dtype=torch.bool, device=device)
        m[self.WORD_OFFSET : self.WORD_OFFSET + self.V] = True
        return m

    # -- feedback --------------------------------------------------------
    def feedback_tokens(self, fb: str) -> list[int]:
        return [self.fb_to_token[c] for c in fb]

    def encode_prefix(self, guesses: list[str], feedbacks: list[str]) -> list[int]:
        """Token stream for a game state: START, then (word, fb x5) per turn."""
        toks = [self.START]
        for g, fb in zip(guesses, feedbacks):
            toks.append(self.word_id(g))
            toks.extend(self.feedback_tokens(fb))
        return toks

    def state_size(self) -> int:
        """Max tokens for a finished game: START + 6 * (1 word + 5 feedback)."""
        return 1 + 6 * 6

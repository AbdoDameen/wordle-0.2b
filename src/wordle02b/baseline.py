"""Greedy entropy baseline solver.

At every turn it picks the guess (from a sampled pool of legal words) that
maximizes the expected information of the feedback, then filters the candidate
answers by the actual feedback. Strong enough to win ~96% of games on the
official answer list; this is the "teacher" that generates training data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .game import WordleGame
from .words import Vocabulary

N_PATTERNS = 3**5  # 243 feedback patterns

# Pattern id packs the five feedback letters as base-3 digits:
# B=0, Y=1, G=2, digit 0 is the first letter (most significant).
_DIGITS = np.array([81, 27, 9, 3, 1], dtype=np.int16)
_BYTES = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", dtype=np.uint8)


def pattern_ids_for_guess(words: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Pattern id of one guess vs every word row in `words` (n, 5) uint8 minus 'a'."""
    greens = words == g[None, :]
    n = words.shape[0]
    cnt = np.zeros((n, 26), dtype=np.int8)
    rows = np.arange(n)
    for i in range(5):
        np.add.at(cnt, (rows, words[:, i]), 1)
    for i in range(5):
        if greens[:, i].any():
            cnt[rows, g[i]] -= greens[:, i]
    res = np.zeros((n, 5), dtype=np.int8)
    res[greens] = 2
    for i in range(5):
        mask = (~greens[:, i]) & (cnt[:, g[i]] > 0)
        res[mask, i] = 1
        cnt[mask, g[i]] -= 1
    return res @ _DIGITS


def build_pattern_matrix(vocab: Vocabulary) -> np.ndarray:
    """pattern_matrix[guess_word_id, answer_idx] = pattern id (0..242).

    vocab.answers is sorted, so row order of the answer array lines up with
    the matrix columns directly.
    """
    answers = vocab.answers
    words = np.array([list(w.encode()) for w in vocab.words], dtype=np.uint8) - ord("a")
    ans = np.array([list(w.encode()) for w in answers], dtype=np.uint8) - ord("a")
    P = np.zeros((vocab.V, len(answers)), dtype=np.int16)
    for wid, w in enumerate(vocab.words):
        P[wid] = pattern_ids_for_guess(ans, words[wid])
    return P


def build_pattern_matrix_cached(vocab: Vocabulary, cache_dir: Path | None = None) -> np.ndarray:
    """Build the pattern matrix, caching to disk (17s build, 30MB npy)."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"pattern_matrix_v{vocab.V}.npy"
        if path.exists():
            P = np.load(path)
            if P.shape == (vocab.V, len(vocab.answers)):
                return P
    P = build_pattern_matrix(vocab)
    if cache_dir is not None:
        np.save(path, P)
    return P


def pattern_id_to_string(pid: int) -> str:
    chars = "BYG"
    s = []
    for _ in range(5):
        pid, d = divmod(pid, 3)
        s.append(chars[d])
    return "".join(reversed(s))


def make_answer_index_array(vocab: Vocabulary) -> np.ndarray:
    """answer_index_array[word_id] = index into vocab.answers, or -1."""
    arr = np.full(vocab.V, -1, dtype=np.int64)
    arr[np.asarray(vocab.answer_ids, dtype=np.int64)] = np.arange(len(vocab.answers))
    return arr


def filter_candidates(
    pattern_matrix: np.ndarray,
    answer_index: np.ndarray,
    cand_word_ids: np.ndarray,
    guess_id: int,
    pid: int,
) -> np.ndarray:
    """Keep candidates consistent with the observed pattern."""
    col = answer_index[cand_word_ids]
    return cand_word_ids[pattern_matrix[guess_id, col] == pid]


def guess_entropies(
    pattern_matrix: np.ndarray,
    answer_index: np.ndarray,
    pool: np.ndarray,
    cand_word_ids: np.ndarray,
) -> np.ndarray:
    """Expected information (bits) of each guess in `pool` over the candidates."""
    col = answer_index[cand_word_ids]
    P = pattern_matrix[pool][:, col]  # (len(pool), len(cand))
    n_cand = P.shape[1]
    counts = np.zeros((P.shape[0], N_PATTERNS), dtype=np.int32)
    rows = np.repeat(np.arange(P.shape[0]), n_cand)
    np.add.at(counts, (rows, P.ravel()), 1)
    probs = counts / n_cand
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(probs > 0, probs * np.log2(probs), 0.0), axis=1)
    return ent


class EntropySolver:
    """Plays games using greedy expected-information maximization.

    `explore` (>0) makes data generation non-deterministic: with that
    probability the guess is sampled from the top-`top_k` entropy words
    instead of the argmax. That variety is what lets a student model
    generalize instead of memorizing one trajectory per answer.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        pattern_matrix: np.ndarray,
        pool_size: int = 300,
        explore: float = 0.2,
        top_k: int = 3,
    ):
        self.vocab = vocab
        self.P = pattern_matrix
        self.answer_index = make_answer_index_array(vocab)
        self.pool_size = pool_size
        self.explore = explore
        self.top_k = top_k
        self.answer_ids = np.asarray(vocab.answer_ids, dtype=np.int64)

    def next_guess(self, cand_word_ids: np.ndarray, rng: np.random.Generator | None = None) -> int:
        if len(cand_word_ids) == 1:
            return int(cand_word_ids[0])
        pool = cand_word_ids
        if len(cand_word_ids) > self.pool_size:
            if rng is None:
                pool = cand_word_ids[: self.pool_size]
            else:
                pool = rng.choice(cand_word_ids, self.pool_size, replace=False)
        ent = guess_entropies(self.P, self.answer_index, pool, cand_word_ids)
        order = np.argsort(-ent)
        if rng is None or rng.random() >= self.explore:
            return int(pool[order[0]])
        return int(rng.choice(pool[order[: self.top_k]]))

    def play(
        self,
        secret: str,
        first_guess: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> WordleGame:
        vocab = self.vocab
        game = WordleGame(secret)
        cand_word_ids = self.answer_ids.copy()
        secret_wid = vocab.word_id(secret)
        for _ in range(game.max_guesses):
            if first_guess is not None and not game.guesses:
                guess = first_guess
            else:
                guess = vocab.word(self.next_guess(cand_word_ids, rng))
            fb = game.play(guess)
            if game.solved:
                break
            guess_id = vocab.word_id(guess)
            pid = int(self.P[guess_id, self.answer_index[secret_wid]])
            cand_word_ids = filter_candidates(self.P, self.answer_index, cand_word_ids, guess_id, pid)
        return game

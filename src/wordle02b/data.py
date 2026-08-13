"""Training data generation.

We let the entropy solver (with exploration) play games against random
answers and record every (game state -> next guess) decision. Games are
serialized into one long token stream:

    [START w1 f1x5 w2 f2x5 ... wk fkx5] PAD [START ...]

The training loop slices blocks of `block_size` tokens from this stream,
shifts by one, and only computes loss at positions whose target is a word
token (where the model was supposed to produce a guess).
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .baseline import EntropySolver, filter_candidates, make_answer_index_array
from .game import WordleGame
from .words import Vocabulary


def _game_stream(
    seed: int,
    vocab: Vocabulary,
    secret_ids: np.ndarray,
    P: np.ndarray,
    answer_index: np.ndarray,
    n_games: int,
    random_opener_p: float = 0.3,
    max_guesses: int = 6,
) -> list[int]:
    rng = np.random.default_rng(seed)
    solver = EntropySolver(vocab, P)
    stream: list[int] = []
    for _ in range(n_games):
        secret_wid = int(rng.choice(secret_ids))
        secret = vocab.word(secret_wid)
        game = WordleGame(secret, max_guesses=max_guesses)
        cand_word_ids = secret_ids.copy()
        stream.append(vocab.START)
        for _ in range(max_guesses):
            if not game.guesses and rng.random() < random_opener_p:
                guess_wid = int(rng.choice(secret_ids))
            else:
                guess_wid = solver.next_guess(cand_word_ids, rng)
            fb = game.play(vocab.word(guess_wid))
            stream.append(guess_wid)
            stream.extend(vocab.feedback_tokens(fb))
            if game.solved:
                break
            pid = int(P[guess_wid, answer_index[secret_wid]])
            cand_word_ids = filter_candidates(P, answer_index, cand_word_ids, guess_wid, pid)
        stream.append(vocab.PAD)
    return stream


def generate_data(
    n_games: int,
    vocab: Vocabulary,
    pattern_matrix: np.ndarray,
    seed: int = 0,
    n_workers: int | None = None,
    random_opener_p: float = 0.3,
    secret_ids: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Generate n_games of teacher play; returns the int32 token stream.

    secret_ids: which answers may be chosen as secrets (pass the train split
    from split_answers to keep a holdout clean for evaluation).
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if secret_ids is None:
        secret_ids = np.asarray(vocab.answer_ids, dtype=np.int64)
    secret_ids = np.asarray(secret_ids, dtype=np.int64)
    answer_index = make_answer_index_array(vocab)
    per_worker = max(1, n_games // n_workers)
    args = [
        (seed + i * 7919, vocab, secret_ids, pattern_matrix, answer_index, per_worker, random_opener_p)
        for i in range(n_workers)
    ]
    remainder = n_games - per_worker * n_workers
    if remainder > 0:
        args[0] = (
            seed, vocab, secret_ids, pattern_matrix, answer_index,
            per_worker + remainder, random_opener_p,
        )

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        streams = pool.starmap(_game_stream, args)
    toks = np.concatenate([np.asarray(s, dtype=np.int32) for s in streams])
    print(f"generated {n_games} games -> {len(toks):,} tokens in {time.time() - t0:.1f}s")
    return toks


def cache_path(n_games: int, seed: int, random_opener_p: float, cache_dir: Path, secret_ids=None) -> Path:
    tag = ""
    if secret_ids is not None:
        import hashlib

        tag = "_ho" + hashlib.md5(np.asarray(secret_ids, dtype=np.int64).tobytes()).hexdigest()[:6]
    return cache_dir / f"games_{n_games}_s{seed}_ro{random_opener_p:.2f}{tag}.npy"


def get_or_generate(
    n_games: int,
    vocab: Vocabulary,
    pattern_matrix: np.ndarray,
    cache_dir: Path,
    seed: int = 0,
    random_opener_p: float = 0.3,
    secret_ids: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Generate data, caching to disk so re-runs are instant."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(n_games, seed, random_opener_p, cache_dir, secret_ids)
    if path.exists():
        toks = np.load(path)
        print(f"loaded cached data: {path.name} ({len(toks):,} tokens)")
        return toks
    toks = generate_data(n_games, vocab, pattern_matrix, seed=seed, random_opener_p=random_opener_p, secret_ids=secret_ids)
    np.save(path, toks)
    print(f"cached to {path}")
    return toks

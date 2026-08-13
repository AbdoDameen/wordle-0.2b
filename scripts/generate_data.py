#!/usr/bin/env python3
"""Generate teacher training data and cache it to data/cache/.

Example:
    python scripts/generate_data.py --games 300000 --workers 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wordle02b import Vocabulary, build_pattern_matrix_cached, load_word_lists
from wordle02b.data import get_or_generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300_000, help="teacher games to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None, help="parallel workers (default: cpus-1)")
    ap.add_argument("--random-openers", type=float, default=0.3, help="fraction of games starting with a random word")
    args = ap.parse_args()

    vocab = Vocabulary(*load_word_lists())
    P = build_pattern_matrix_cached(vocab, cache_dir="data/cache")
    toks = get_or_generate(
        args.games, vocab, P,
        cache_dir="data/cache", seed=args.seed,
        random_opener_p=args.random_openers,
    )
    print(f"tokens: {len(toks):,}")


if __name__ == "__main__":
    main()

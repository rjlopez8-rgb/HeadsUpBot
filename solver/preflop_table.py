"""
solver/preflop_table.py - exact-ish preflop equity (vs a random hand) for all
376,992 five-card starting hands, indexed by the colex rank of the sorted
cards. Used by the bot at runtime (no Monte Carlo preflop) and by the dataset.

    python solver/preflop_table.py --samples 512     -> data/preflop_eq.npy (float32)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from itertools import combinations
from math import comb

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(_ROOT, "data", "preflop_eq.npy")


def all_hands() -> np.ndarray:
    return np.array(list(combinations(range(36), 5)), dtype=np.int64)   # lexicographic


def _chunk(args):
    import fasteval as fe
    from dataset import batch_equity
    lo, hi, k, seed = args
    fe.load()
    rng = np.random.default_rng(seed)
    H = all_hands()[lo:hi]
    out = np.zeros(hi - lo, dtype=np.float32)
    dummy = np.zeros((H.shape[0], 5), dtype=np.int64)
    for i in range(0, H.shape[0], 150):
        out[i:i + 150] = batch_equity(H[i:i + 150], dummy[i:i + 150], 0, k, rng)
    return lo, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=512)
    a = ap.parse_args()
    H = all_hands()
    n = H.shape[0]
    t = time.time()
    step = 6000
    jobs = [(lo, min(n, lo + step), a.samples, 1000 + lo) for lo in range(0, n, step)]
    lex = np.zeros(n, dtype=np.float32)
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        for lo, out in pool.imap_unordered(_chunk, jobs):
            lex[lo:lo + len(out)] = out
    # re-index by colex rank so lookup needs no search
    C = np.array([[comb(c, k) for k in range(6)] for c in range(36)], dtype=np.int64)
    colex = C[H[:, 0], 1] + C[H[:, 1], 2] + C[H[:, 2], 3] + C[H[:, 3], 4] + C[H[:, 4], 5]
    table = np.zeros(n, dtype=np.float32)
    table[colex] = lex
    np.save(OUT, table)
    print(f"{n} hands x {a.samples} samples in {time.time() - t:.0f}s -> {OUT}; "
          f"min {table.min():.3f} max {table.max():.3f} mean {table.mean():.3f}")

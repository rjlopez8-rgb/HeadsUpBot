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


def canonical_classes(H: np.ndarray):
    """Suit-isomorphism classes: for each hand, the smallest colex rank over
    all 24 suit relabelings. Returns (class id per hand, representative hands)."""
    from itertools import permutations
    C = np.array([[comb(c, k) for k in range(6)] for c in range(36)], dtype=np.int64)
    best = None
    for perm in permutations(range(4)):
        pm = np.array(perm)
        mapped = (H >> 2) * 4 + pm[H & 3]
        mapped.sort(axis=1)
        cx = (C[mapped[:, 0], 1] + C[mapped[:, 1], 2] + C[mapped[:, 2], 3]
              + C[mapped[:, 3], 4] + C[mapped[:, 4], 5])
        best = cx if best is None else np.minimum(best, cx)
    uniq, first, inv = np.unique(best, return_index=True, return_inverse=True)
    return inv, H[first]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--canonical", action="store_true",
                    help="compute once per suit-isomorphism class (~16k) with many more samples")
    a = ap.parse_args()
    if a.canonical:
        H = all_hands()
        t = time.time()
        inv, reps = canonical_classes(H)
        print(f"{len(reps)} suit classes", flush=True)
        k = a.samples
        jobs = [(reps[i:i + 400], k, 77 + i) for i in range(0, len(reps), 400)]

        def run(job):
            import fasteval as fe
            from dataset import batch_equity
            hands, kk, seed = job
            fe.load()
            rng = np.random.default_rng(seed)
            out = np.zeros(len(hands), dtype=np.float32)
            dummy = np.zeros((len(hands), 5), dtype=np.int64)
            step = max(1, 40000 // kk)
            for i in range(0, len(hands), step):
                out[i:i + step] = batch_equity(hands[i:i + step], dummy[i:i + step], 0, kk, rng)
            return out
        with mp.get_context("fork").Pool(int(os.environ.get("PROCS", os.cpu_count()))) as pool:
            cls_eq = np.concatenate(pool.map(run, jobs))
        lex = cls_eq[inv]
        C = np.array([[comb(c, kk) for kk in range(6)] for c in range(36)], dtype=np.int64)
        colex = C[H[:, 0], 1] + C[H[:, 1], 2] + C[H[:, 2], 3] + C[H[:, 3], 4] + C[H[:, 4], 5]
        table = np.zeros(len(H), dtype=np.float32)
        table[colex] = lex
        np.save(OUT, table)
        print(f"done in {time.time() - t:.0f}s -> {OUT}; mean {table.mean():.4f}")
        sys.exit(0)
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

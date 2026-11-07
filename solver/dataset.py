"""
solver/dataset.py - deal many hands and record, for both players on every
street, the hand-strength feature the bot will see at runtime, plus who
actually wins at showdown.

Feature = preflop: table equity (data/preflop_eq.npy); later streets: Monte Carlo equity vs a random opponent hand (and random runout),
computed with exactly the same estimator the bot uses at runtime
(pokerlib-style sampling, K_SAMPLES samples), so training and play match.

Output (solver/data/deals_<tag>.npz):
    eq      float32 (N, 2, 4)   equity of P0 (button) / P1 on pre, flop, turn, river
    outcome int8    (N,)        +1 P0 wins, -1 P1 wins, 0 split

    python solver/dataset.py --n 300000 --tag a
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import fasteval as fe  # noqa: E402

K_SAMPLES = (128, 128, 128, 160)     # MC samples per street (pre, flop, turn, river)
BOARD_N = (0, 3, 4, 5)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def batch_equity(holes: np.ndarray, boards: np.ndarray, nb: int, k: int, rng) -> np.ndarray:
    """Equity of each of D hands (D,5) on the first nb cards of boards (D,5)
    vs a random opponent hand + random runout, k samples each."""
    D = holes.shape[0]
    known = np.concatenate([holes, boards[:, :nb]], axis=1)          # (D, 5+nb)
    m = 5 + (5 - nb)                                                  # opp hole + runout
    keys = rng.random((D, k, 36), dtype=np.float32)
    rows = np.arange(D)[:, None]
    blocked = np.zeros((D, 36), dtype=bool)
    blocked[rows, known] = True
    keys[np.broadcast_to(blocked[:, None, :], keys.shape)] = 2.0
    draw = np.argpartition(keys, m - 1, axis=2)[:, :, :m].astype(np.int64)   # (D,k,m)
    opp = draw[:, :, :5].reshape(-1, 5)
    full_board = np.concatenate(
        [np.broadcast_to(boards[:, None, :nb], (D, k, nb)), draw[:, :, 5:]], axis=2).reshape(-1, 5)
    mine = np.broadcast_to(holes[:, None, :], (D, k, 5)).reshape(-1, 5)
    a = fe.omaha_batch(mine, full_board)
    b = fe.omaha_batch(opp, full_board)
    w = (a > b) + 0.5 * (a == b)
    return w.reshape(D, k).mean(axis=1)


def _chunk(args):
    seed, n = args
    fe.load()
    rng = np.random.default_rng(seed)
    keys = rng.random((n, 36))
    cards = np.argpartition(keys, 14, axis=1)[:, :15].astype(np.int64)
    h0, h1, board = cards[:, 0:5], cards[:, 5:10], cards[:, 10:15]
    eq = np.zeros((n, 2, 4), dtype=np.float32)
    import pokerlib as pl
    step = 400
    for p, h in enumerate((h0, h1)):
        eq[:, p, 0] = pl.preflop_eq(h)                    # same table as the bot uses
    for i in range(0, n, step):
        sl = slice(i, min(n, i + step))
        for p, h in enumerate((h0, h1)):
            for s in range(1, 4):
                eq[sl, p, s] = batch_equity(h[sl], board[sl], BOARD_N[s], K_SAMPLES[s], rng)
    s0 = fe.omaha_batch(h0, board)
    s1 = fe.omaha_batch(h1, board)
    outcome = np.sign(s0.astype(np.int64) - s1.astype(np.int64)).astype(np.int8)
    return eq, outcome, cards.astype(np.int8)


def generate(n: int, seed: int, procs: int) -> tuple[np.ndarray, np.ndarray]:
    per = 10000
    jobs = [(seed * 100003 + i, min(per, n - i * per)) for i in range((n + per - 1) // per)]
    with mp.get_context("fork").Pool(procs) as pool:
        parts = pool.map(_chunk, jobs, chunksize=1)
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    t = time.time()
    eq, oc, cards = generate(a.n, a.seed, a.procs)
    np.savez_compressed(os.path.join(DATA, f"deals_{a.tag}.npz"), eq=eq, outcome=oc, cards=cards)
    print(f"{a.n} deals in {time.time() - t:.0f}s; mean eq by street:", eq.mean(axis=(0, 1)).round(3),
          "P0 win/split/lose:", [(oc == v).mean().round(3) for v in (1, 0, -1)])

"""
solver/features.py - potential-aware hand strength: E[HS^2].

HS on the river = chance to beat a random opponent hand. Before the river,
E[HS] averages it over runouts, which lumps together "medium now" and
"big draw: great or terrible". E[HS^2] (average of the SQUARED river
strength over runouts) separates them: draws get higher E[HS^2] than made
hands of the same E[HS]. Bucketing on E[HS^2] is the classic improvement
used by poker AIs (e.g. the CPRG's bots).

Estimator (same at training and runtime): R runouts x O opponent hands.
    python solver/features.py --tags c d e f      # writes deals_<tag>_h2.npz
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import fasteval as fe  # noqa: E402

R_RUNOUTS = 16
O_OPPS = 16
DATA = os.path.join(_HERE, "data")


def batch_ehs2(holes: np.ndarray, boards: np.ndarray, nb: int, rng, R: int = R_RUNOUTS,
               O: int = O_OPPS) -> tuple[np.ndarray, np.ndarray]:
    """(E[HS], E[HS^2]) for D hands on the first nb (3 or 4) board cards."""
    D = holes.shape[0]
    known = np.concatenate([holes, boards[:, :nb]], axis=1)
    need = 5 - nb
    rows = np.arange(D)[:, None]
    blocked = np.zeros((D, 36), dtype=bool)
    blocked[rows, known] = True
    k1 = rng.random((D, R, 36), dtype=np.float32)
    k1[np.broadcast_to(blocked[:, None, :], k1.shape)] = 2.0
    run = np.argpartition(k1, need - 1, axis=2)[:, :, :need].astype(np.int64)        # (D,R,need)
    blocked2 = np.broadcast_to(blocked[:, None, :], (D, R, 36)).copy()
    np.put_along_axis(blocked2, run, True, axis=2)
    k2 = rng.random((D, R, O, 36), dtype=np.float32)
    k2[np.broadcast_to(blocked2[:, :, None, :], k2.shape)] = 2.0
    opp = np.argpartition(k2, 4, axis=3)[:, :, :, :5].astype(np.int64)               # (D,R,O,5)
    board_full = np.concatenate([np.broadcast_to(boards[:, None, :nb], (D, R, nb)), run], axis=2)  # (D,R,5)
    mine = fe.omaha_batch(np.broadcast_to(holes[:, None, :], (D, R, 5)).reshape(-1, 5),
                          board_full.reshape(-1, 5)).reshape(D, R)
    theirs = fe.omaha_batch(opp.reshape(-1, 5),
                            np.broadcast_to(board_full[:, :, None, :], (D, R, O, 5)).reshape(-1, 5)
                            ).reshape(D, R, O)
    w = (mine[:, :, None] > theirs) + 0.5 * (mine[:, :, None] == theirs)
    hs = w.mean(axis=2)                                                               # (D,R)
    return hs.mean(axis=1), (hs ** 2).mean(axis=1)


def _chunk(args):
    cards, seed = args
    fe.load()
    rng = np.random.default_rng(seed)
    cards = cards.astype(np.int64)
    n = cards.shape[0]
    out = np.zeros((n, 2, 2), dtype=np.float32)          # player, street(flop, turn) -> E[HS^2]
    board = cards[:, 10:15]
    for i in range(0, n, 150):
        sl = slice(i, min(n, i + 150))
        for p in range(2):
            h = cards[sl, 5 * p:5 * p + 5]
            for j, nb in enumerate((3, 4)):
                out[sl, p, j] = batch_ehs2(h, board[sl], nb, rng)[1]
    return out


def convert(tag: str, procs: int):
    z = np.load(os.path.join(DATA, f"deals_{tag}.npz"))
    cards, eq, oc = z["cards"], z["eq"].copy(), z["outcome"]
    parts = np.array_split(cards, max(1, len(cards) // 5000))
    with mp.get_context("fork").Pool(procs) as pool:
        res = pool.map(_chunk, [(p, 7919 * i + hash(tag) % 1000) for i, p in enumerate(parts)])
    h2 = np.concatenate(res)
    eq[:, :, 1] = h2[:, :, 0]
    eq[:, :, 2] = h2[:, :, 1]
    eq[:, :, 3] = eq[:, :, 3] ** 2            # river: HS^2 (monotone in HS, same buckets)
    np.savez_compressed(os.path.join(DATA, f"deals_{tag}_h2.npz"), eq=eq, outcome=oc, cards=cards)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    for t in a.tags:
        t0 = time.time()
        convert(t, a.procs)
        print(f"{t}: E[HS^2] features in {time.time() - t0:.0f}s", flush=True)

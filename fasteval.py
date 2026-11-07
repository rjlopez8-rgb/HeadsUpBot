"""
fasteval.py - vectorized (numpy) Omaha hand scoring, many hands at once.

The provided evaluator.omaha_score scores ONE hand in ~27 microseconds
(100 dict lookups in pure Python). Monte Carlo needs thousands of scores per
decision, so this module scores a whole batch of hands with a few numpy ops.

Idea: a 2-hole-card + 3-board-card hand is identified by
    (index of the hole PAIR among C(36,2)=630) x (index of the board TRIPLE among C(36,3)=7140)
so PT_TABLE[pair_idx * 7140 + triple_idx] is the score of that 5-card hand,
with no sorting needed. An Omaha hand is the max over 10 pairs x 10 triples.

The table (4.5M int32 = 18 MB) is built from evaluator.py's own scores, so it
agrees with the judge exactly. It is cached to data/pt_table.npy (built in
~2-4 s if the file is missing).
"""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np

N_CARDS = 36
N_PAIRS = 630
N_TRIPLES = 7140

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE_PATH = os.path.join(_HERE, "data", "pt_table.npy")

PAIRS5 = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
TRIPLES5 = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # (10, 3)

# PAIR_IDX[a, b] / TRIPLE_IDX[a, b, c]: index of an unordered card set (any order).
PAIR_IDX = np.full((N_CARDS, N_CARDS), -1, dtype=np.int64)
_pairs = list(combinations(range(N_CARDS), 2))
for _i, (_a, _b) in enumerate(_pairs):
    PAIR_IDX[_a, _b] = PAIR_IDX[_b, _a] = _i
TRIPLE_IDX = np.full((N_CARDS, N_CARDS, N_CARDS), -1, dtype=np.int64)
_triples = list(combinations(range(N_CARDS), 3))
for _i, (_a, _b, _c) in enumerate(_triples):
    for _p in ((_a, _b, _c), (_a, _c, _b), (_b, _a, _c), (_b, _c, _a), (_c, _a, _b), (_c, _b, _a)):
        TRIPLE_IDX[_p] = _i

PT_TABLE: np.ndarray | None = None


def _build_table() -> np.ndarray:
    from evaluator import get_table
    t5 = get_table()  # 36-bit mask -> score
    pair_masks = np.array([(1 << a) | (1 << b) for a, b in _pairs], dtype=np.int64)
    trip_masks = np.array([(1 << a) | (1 << b) | (1 << c) for a, b, c in _triples], dtype=np.int64)
    # All 5-card masks in a sorted array, with their scores, for a vectorized lookup.
    keys = np.fromiter(t5.keys(), dtype=np.int64, count=len(t5))
    vals = np.fromiter(t5.values(), dtype=np.int64, count=len(t5))
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]
    table = np.zeros(N_PAIRS * N_TRIPLES, dtype=np.int32)
    for p in range(N_PAIRS):
        m = pair_masks[p] | trip_masks
        ok = (pair_masks[p] & trip_masks) == 0
        pos = np.searchsorted(keys, m[ok])
        row = np.zeros(N_TRIPLES, dtype=np.int32)
        row[ok] = vals[pos]
        table[p * N_TRIPLES:(p + 1) * N_TRIPLES] = row
    return table


def load() -> np.ndarray:
    """Load (or build) the pair x triple score table. Call once at startup."""
    global PT_TABLE
    if PT_TABLE is None:
        try:
            PT_TABLE = np.load(_TABLE_PATH)
        except Exception:
            PT_TABLE = _build_table()
    return PT_TABLE


def save() -> None:
    os.makedirs(os.path.dirname(_TABLE_PATH), exist_ok=True)
    np.save(_TABLE_PATH, _build_table())


def hole_pair_idx(holes: np.ndarray) -> np.ndarray:
    """holes (..., 5) -> (..., 10) pair indices."""
    return PAIR_IDX[holes[..., PAIRS5[:, 0]], holes[..., PAIRS5[:, 1]]]


def board_triple_idx(boards: np.ndarray) -> np.ndarray:
    """boards (..., 5) -> (..., 10) triple indices."""
    return TRIPLE_IDX[boards[..., TRIPLES5[:, 0]], boards[..., TRIPLES5[:, 1]], boards[..., TRIPLES5[:, 2]]]


def omaha_batch(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Best Omaha scores for N hands. holes (N,5) or (5,), boards (N,5)."""
    t = PT_TABLE if PT_TABLE is not None else load()
    p = hole_pair_idx(np.asarray(holes))        # (N,10) or (10,)
    q = board_triple_idx(np.asarray(boards))    # (N,10)
    if p.ndim == 1:
        p = p[None, :]
    idx = p[:, :, None] * N_TRIPLES + q[:, None, :]
    return t[idx].max(axis=(1, 2))


if __name__ == "__main__":
    save()
    print("saved", _TABLE_PATH)

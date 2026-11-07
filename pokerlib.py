"""
pokerlib.py - shared poker math for our bot and the lab's opponent bots.

All the heavy lifting is vectorized with numpy on top of fasteval.py.

    equity(hole, board, n, rng, keep=1.0)
        Monte Carlo chance to win (ties count half) against an opponent whose
        hand is sampled at random, optionally restricted to the strongest
        `keep` fraction of opponent hands (a simple "range" model: if the
        opponent only bets with its best 30% of hands, use keep=0.3).

    preflop_strength(holes)
        Cheap vectorized 0..1 score of 5-card starting hands (used to rank
        sampled opponent hands preflop). A precomputed equity table can
        replace it later.
"""

from __future__ import annotations

import numpy as np

import fasteval as fe
from fasteval import N_TRIPLES, PAIR_IDX, PAIRS5, TRIPLE_IDX

fe.load()

ALL_CARDS = np.arange(36, dtype=np.int64)

# Preflop equity vs a random hand for every 5-card hand, indexed by the colex
# rank of the sorted cards (solver/preflop_table.py). None if not built yet.
from math import comb as _comb  # noqa: E402
import os as _os  # noqa: E402
_COMB = np.array([[_comb(c, k) for k in range(6)] for c in range(36)], dtype=np.int64)
try:
    PREFLOP_EQ = np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "preflop_eq.npy"))
except Exception:  # noqa: BLE001
    PREFLOP_EQ = None


def colex(holes: np.ndarray) -> np.ndarray:
    """Colex rank of (N,5) or (5,) card sets (any order)."""
    h = np.sort(np.asarray(holes, dtype=np.int64), axis=-1)
    return (_COMB[h[..., 0], 1] + _COMB[h[..., 1], 2] + _COMB[h[..., 2], 3]
            + _COMB[h[..., 3], 4] + _COMB[h[..., 4], 5])


def preflop_eq(holes) -> np.ndarray | float:
    """Table equity vs a random hand (vectorized). Requires PREFLOP_EQ."""
    return PREFLOP_EQ[colex(holes)]


def _made_strength(holes: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Best made-hand score of each hole (N,5) on a PARTIAL board of 3 or 4
    cards (2 hole + 3 board, Omaha rule). Ignores draws. Used for ranking."""
    from itertools import combinations
    trips = np.array(list(combinations(range(len(board)), 3)), dtype=np.int64)
    q = TRIPLE_IDX[board[trips[:, 0]], board[trips[:, 1]], board[trips[:, 2]]]  # (T,)
    p = PAIR_IDX[holes[:, PAIRS5[:, 0]], holes[:, PAIRS5[:, 1]]]                # (N,10)
    idx = p[:, :, None] * N_TRIPLES + q[None, None, :]
    return fe.PT_TABLE[idx].max(axis=(1, 2)).astype(np.float64)


def preflop_strength(holes: np.ndarray) -> np.ndarray:
    """Preflop strength for ranking opponent hands: the equity table when
    available, otherwise a heuristic."""
    if PREFLOP_EQ is not None:
        return PREFLOP_EQ[colex(holes)].astype(np.float64)
    return _preflop_heuristic(holes)


def _preflop_heuristic(holes: np.ndarray) -> np.ndarray:
    """Heuristic 0..~1 strength for (N,5) hole cards. Rewards high cards,
    pairs (especially high), double-suitedness and connectedness."""
    holes = np.asarray(holes)
    ranks = holes >> 2
    suits = holes & 3
    rc = np.zeros((holes.shape[0], 9), dtype=np.int64)
    np.add.at(rc, (np.arange(holes.shape[0])[:, None], ranks), 1)
    sc = np.zeros((holes.shape[0], 4), dtype=np.int64)
    np.add.at(sc, (np.arange(holes.shape[0])[:, None], suits), 1)
    high = (ranks.astype(np.float64) / 8.0).mean(axis=1)                   # 0..1
    pair_ranks = (rc >= 2) * (np.arange(9) + 1)
    best_pair = pair_ranks.max(axis=1) / 9.0                                # 0..1
    n_pairs = (rc >= 2).sum(axis=1)
    trips = (rc >= 3).any(axis=1)                                           # dead cards
    suited2 = (sc >= 2).sum(axis=1)                                         # # of suits with 2+
    over_suited = (sc >= 4).any(axis=1)
    present = (rc > 0).astype(np.int64)
    # connectedness: longest window of 5 consecutive ranks covered (ace low too)
    ext = np.concatenate([present[:, 8:9], present], axis=1)                # A,6..A
    win = np.stack([ext[:, i:i + 5].sum(axis=1) for i in range(6)], axis=1).max(axis=1)
    s = (0.9 * high + 0.55 * best_pair + 0.12 * np.minimum(n_pairs, 2)
         + 0.18 * np.minimum(suited2, 2) + 0.10 * (win - 2).clip(0) / 3.0
         - 0.25 * trips - 0.12 * over_suited)
    return s / 2.0


def sample_cards(rng: np.random.Generator, deck: np.ndarray, n: int, k: int) -> np.ndarray:
    """n independent draws of k distinct cards from deck -> (n, k)."""
    keys = rng.random((n, deck.shape[0]))
    return deck[np.argpartition(keys, k - 1, axis=1)[:, :k]]


def equity(hole, board, n: int, rng: np.random.Generator, keep: float = 1.0,
           oversample: int = 3) -> float:
    """Monte Carlo equity vs a random (or top-`keep` fraction) opponent hand."""
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    if board.shape[0] == 0 and keep >= 0.999 and PREFLOP_EQ is not None:
        return float(PREFLOP_EQ[colex(hole)])          # exact table, no sampling
    dead = np.zeros(36, dtype=bool)
    dead[hole] = True
    dead[board] = True
    deck = ALL_CARDS[~dead]
    need = 5 - board.shape[0]
    keep = float(min(1.0, max(0.02, keep)))
    m = n if keep >= 0.999 else int(min(n * oversample, max(n, n / keep)))
    draw = sample_cards(rng, deck, m, 5 + need)
    opp = draw[:, :5]
    if keep < 0.999:
        if board.shape[0] == 0:
            st = preflop_strength(opp)
        else:
            st = _made_strength(opp, board) + 0.5 * preflop_strength(opp)  # break exact ties only
        k = max(8, int(round(m * keep)))
        top = np.argpartition(-st, k - 1)[:k]
        draw = draw[top]
        opp = draw[:, :5]
    boards = np.concatenate([np.broadcast_to(board, (draw.shape[0], board.shape[0])), draw[:, 5:]], axis=1)
    mine = fe.omaha_batch(hole, boards)
    theirs = fe.omaha_batch(opp, boards)
    return float(((mine > theirs) + 0.5 * (mine == theirs)).mean())

"""
solver/abstraction2.py - 2D ("strength x potential") hand buckets for flop/turn.

For each hand on the flop and turn we have E[HS] (average final strength) and
E[HS^2]; their difference gives the spread of final strength over runouts:
    sd = sqrt(E[HS^2] - E[HS]^2)   (0 for a made hand that won't change,
                                     large for a draw that hits or misses)
k-means on (E[HS], sd) puts hands with the same average but very different
potential into different buckets. Preflop (table equity) and river (HS) keep
equal-mass 1D buckets.

    python solver/abstraction2.py --tags c d e f g --K 20 --out solver/data/buckets2d_K20.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SD_WEIGHT = 1.0


def features(ehs: np.ndarray, ehs2: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.clip(ehs2 - ehs * ehs, 0, None))
    return np.stack([ehs, SD_WEIGHT * sd], axis=-1)


def kmeans(x: np.ndarray, K: int, iters: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # k-means++ init
    c = [x[rng.integers(len(x))]]
    for _ in range(1, K):
        d = np.min(((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
        c.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(c)
    for _ in range(iters):
        lab = assign(x, c)
        newc = np.array([x[lab == k].mean(axis=0) if (lab == k).any() else c[k] for k in range(K)])
        if np.allclose(newc, c, atol=1e-6):
            break
        c = newc
    order = np.lexsort((c[:, 1], c[:, 0]))          # sort buckets by strength, then spread
    return c[order]


def assign(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=np.int64)
    for i in range(0, len(x), 200000):
        out[i:i + 200000] = (((x[i:i + 200000, None, :] - c[None]) ** 2).sum(-1)).argmin(axis=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # both moments from the SAME runouts (solver/features.py, FEATURES_SUFFIX=h2b)
    h2 = [np.load(os.path.join(DATA, f"deals_{t}_h2b.npz")) for t in a.tags]
    eq = np.concatenate([b["ehs"] for b in h2])           # E[HS]: table pre, same-run flop/turn, MC river
    eq2 = np.concatenate([b["eq"] for b in h2])           # E[HS^2] on flop/turn
    oc = np.concatenate([b["outcome"] for b in h2])
    N = len(eq)
    buck = np.zeros((N, 2, 4), dtype=np.int64)
    thr = np.zeros((4, a.K - 1))
    cents = np.zeros((4, a.K, 2))
    for s in (0, 3):
        thr[s] = np.quantile(eq[:, :, s].ravel(), np.linspace(0, 1, a.K + 1)[1:-1])
        buck[:, :, s] = np.searchsorted(thr[s], eq[:, :, s], side="right")
    rng = np.random.default_rng(1)
    for s in (1, 2):
        x = features(eq[:, :, s].ravel().astype(np.float64), eq2[:, :, s].ravel().astype(np.float64))
        sample = x[rng.choice(len(x), size=min(300000, len(x)), replace=False)]
        cents[s] = kmeans(sample, a.K, seed=s)
        buck[:, :, s] = assign(x, cents[s]).reshape(N, 2)
        sizes = np.bincount(buck[:, :, s].ravel(), minlength=a.K) / (2 * N)
        print(f"street {s}: bucket sizes min {sizes.min():.3f} max {sizes.max():.3f}")
        print("   centroids (E[HS], spread):", [tuple(round(v, 2) for v in cc) for cc in cents[s]])
    np.savez_compressed(a.out, buckets=buck.astype(np.int8), thr=thr, centroids=cents, outcome=oc, K=a.K)
    print("saved", a.out)

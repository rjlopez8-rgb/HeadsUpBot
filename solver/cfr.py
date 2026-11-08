"""
solver/cfr.py - vectorized CFR+ for heads-up short deck PLO5 (bucketed).

Abstraction
  * Actions: NONE needed - the full betting tree (solver/tree.py) is solved.
  * Cards: each player's hand on each street is put in one of K equal-mass
    buckets of "equity vs a random hand" (solver/dataset.py). The pair of
    buckets (b0, b1) is the chance state; it moves street to street with
    empirical joint transition matrices T_s (which keep the correlation the
    shared board creates), and showdown pays E[outcome | pair state] taken
    straight from the data (for all-ins, from the street the money went in).
  * A player's decision depends on (its current bucket, betting history).

Algorithm: CFR+ (regret matching+, alternating updates, linear averaging),
processed level by level so every node of a tree depth is one numpy op.
Exploitability (best response in the abstract game) tracks convergence.

    python solver/cfr.py --data a --K 16 --iters 600
Writes solver/data/strategy_K<K>.npz (thresholds + average strategy per node).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from tree import CHANCE, DECISION, FOLD, SHOWDOWN, build  # noqa: E402

DATA = os.path.join(_HERE, "data")


# --------------------------------------------------------------- abstraction
def build_abstraction(eq: np.ndarray, outcome: np.ndarray, K: int, alpha: float = 4.0):
    """Bucket thresholds, initial pair distribution, transitions, outcomes."""
    N = eq.shape[0]
    S = K * K
    thr = np.zeros((4, K - 1))
    pair = np.zeros((4, N), dtype=np.int64)
    buck = np.zeros((4, N, 2), dtype=np.int64)
    for s in range(4):
        thr[s] = np.quantile(eq[:, :, s].ravel(), np.linspace(0, 1, K + 1)[1:-1])
        buck[s] = np.searchsorted(thr[s], eq[:, :, s], side="right")
        pair[s] = buck[s][:, 0] * K + buck[s][:, 1]
    init = np.bincount(pair[0], minlength=S).astype(np.float64) / N
    T = []
    for s in range(3):
        cnt = np.zeros((S, S))
        np.add.at(cnt, (pair[s], pair[s + 1]), 1.0)
        # prior: each player's bucket moves independently (well estimated)
        m = np.zeros((K, K))
        np.add.at(m, (buck[s][:, 0], buck[s + 1][:, 0]), 1.0)
        np.add.at(m, (buck[s][:, 1], buck[s + 1][:, 1]), 1.0)
        m = (m + 1e-3) / (m + 1e-3).sum(axis=1, keepdims=True)
        prior = np.einsum("ac,bd->abcd", m, m).reshape(S, S)
        rows = cnt.sum(axis=1, keepdims=True)
        T.append((cnt + alpha * prior) / (rows + alpha))
    outc = np.zeros((4, S))
    oc = outcome.astype(np.float64)
    for s in range(4):
        tot = np.bincount(pair[s], weights=oc, minlength=S)
        n = np.bincount(pair[s], minlength=S)
        outc[s] = tot / (n + 2.0)                 # shrink sparse states toward 0
    return dict(K=K, thr=thr, init=init, T=T, outc=outc)


def abstraction_from_buckets(buck: np.ndarray, outcome: np.ndarray, K: int, alpha: float = 4.0, **extra):
    """Same as build_abstraction, but with precomputed buckets (N, 2, 4)."""
    N = buck.shape[0]
    S = K * K
    buck = buck.astype(np.int64)
    pair = np.stack([buck[:, 0, s] * K + buck[:, 1, s] for s in range(4)])
    init = np.bincount(pair[0], minlength=S).astype(np.float64) / N
    T = []
    for s in range(3):
        cnt = np.zeros((S, S))
        np.add.at(cnt, (pair[s], pair[s + 1]), 1.0)
        m = np.zeros((K, K))
        np.add.at(m, (buck[:, 0, s], buck[:, 0, s + 1]), 1.0)
        np.add.at(m, (buck[:, 1, s], buck[:, 1, s + 1]), 1.0)
        m = (m + 1e-3) / (m + 1e-3).sum(axis=1, keepdims=True)
        prior = np.einsum("ac,bd->abcd", m, m).reshape(S, S)
        rows = cnt.sum(axis=1, keepdims=True)
        T.append((cnt + alpha * prior) / (rows + alpha))
    outc = np.zeros((4, S))
    oc = outcome.astype(np.float64)
    for s in range(4):
        tot = np.bincount(pair[s], weights=oc, minlength=S)
        n = np.bincount(pair[s], minlength=S)
        outc[s] = tot / (n + 2.0)
    return dict(K=K, init=init, T=T, outc=outc, **extra)


# ---------------------------------------------------------------------- CFR
class Solver:
    def __init__(self, abst: dict, dtype=np.float64):
        self.ab = abst
        K = self.K = abst["K"]
        S = self.S = K * K
        self.dt = dtype
        nodes = self.nodes = build()
        N = self.N = len(nodes)
        self.kind = np.array([n.kind for n in nodes])
        self.street = np.array([n.street for n in nodes])
        self.player = np.array([n.player for n in nodes])
        self.parent = np.array([n.parent for n in nodes])
        depth = np.zeros(N, dtype=np.int64)
        for i, n in enumerate(nodes):
            for c in n.children:
                depth[c] = depth[i] + 1
        self.levels = [np.where(depth == d)[0] for d in range(depth.max() + 1)]
        # decision node bookkeeping
        self.dec = np.where(self.kind == DECISION)[0]
        self.dec_id = -np.ones(N, dtype=np.int64)
        self.dec_id[self.dec] = np.arange(len(self.dec))
        D = len(self.dec)
        self.child = -np.ones((D, 4), dtype=np.int64)        # by action int
        self.legal = np.zeros((D, 4), dtype=bool)
        self.in_action = -np.ones(N, dtype=np.int64)          # action that led here
        for j, i in enumerate(self.dec):
            for a, c in zip(nodes[i].actions, nodes[i].children):
                self.child[j, int(a)] = c
                self.legal[j, int(a)] = True
                self.in_action[c] = int(a)
        # terminal values (P0 perspective), per street outcome
        self.term_const = np.zeros(N)
        for i, n in enumerate(nodes):
            if n.kind == FOLD:
                self.term_const[i] = n.c1 if n.folder == 1 else -n.c0
            elif n.kind == SHOWDOWN:
                self.term_const[i] = n.c0
        self.regret = np.zeros((D, K, 4), dtype=dtype)
        self.avg = np.zeros((D, K, 4), dtype=dtype)
        self.A0 = np.zeros((N, S), dtype=dtype)
        self.A1 = np.zeros((N, S), dtype=dtype)
        self.V = np.zeros((N, S), dtype=dtype)
        self.T = [t.astype(dtype) for t in abst["T"]]
        self.outc = abst["outc"].astype(dtype)
        self.init = abst["init"].astype(dtype)
        self.t = 0

    # strategy from regrets (regret matching+)
    def current(self, which="regret") -> np.ndarray:
        r = self.regret if which == "regret" else self.avg
        pos = np.maximum(r, 0) * self.legal[:, None, :]
        tot = pos.sum(axis=2, keepdims=True)
        uni = self.legal[:, None, :] / self.legal.sum(axis=1)[:, None, None]
        return np.where(tot > 0, pos / np.where(tot > 0, tot, 1), uni).astype(self.dt)

    def _forward(self, sig: np.ndarray, only=None):
        """Reach arrays. only=i: multiply only by player i's strategy for A_i,
        (used by best response where the BR player's reach is irrelevant)."""
        K, S = self.K, self.S
        self.A0[0] = self.init
        self.A1[0] = self.init
        for lev in self.levels[1:]:
            par = self.parent[lev]
            pk = self.kind[par]
            # children of chance nodes: new street
            m = pk == CHANCE
            if m.any():
                c, p = lev[m], par[m]
                for s in range(3):
                    mm = self.street[p] == s
                    if mm.any():
                        self.A0[c[mm]] = self.A0[p[mm]] @ self.T[s]
                        self.A1[c[mm]] = self.A1[p[mm]] @ self.T[s]
            m = pk == DECISION
            if m.any():
                c, p = lev[m], par[m]
                a = self.in_action[c]
                f = sig[self.dec_id[p], :, a]                       # (n, K) for acting player
                pl = self.player[p]
                for who in (0, 1):
                    mm = pl == who
                    if not mm.any():
                        continue
                    cc, pp, ff = c[mm], p[mm], f[mm]
                    if who == 0:
                        self.A0[cc] = (self.A0[pp].reshape(-1, K, K) * ff[:, :, None]).reshape(-1, S)
                        self.A1[cc] = self.A1[pp]
                    else:
                        self.A1[cc] = (self.A1[pp].reshape(-1, K, K) * ff[:, None, :]).reshape(-1, S)
                        self.A0[cc] = self.A0[pp]

    def _terminal_and_chance(self, lev):
        k = self.kind[lev]
        m = k == FOLD
        if m.any():
            self.V[lev[m]] = self.term_const[lev[m]][:, None]
        m = k == SHOWDOWN
        if m.any():
            n = lev[m]
            self.V[n] = self.outc[self.street[n]] * self.term_const[n][:, None]
        m = k == CHANCE
        if m.any():
            n = lev[m]
            ch = np.array([self.nodes[i].children[0] for i in n])
            for s in range(3):
                mm = self.street[n] == s
                if mm.any():
                    self.V[n[mm]] = self.V[ch[mm]] @ self.T[s].T

    def iterate(self, update_player: int):
        K, S = self.K, self.S
        sig = self.current()
        self._forward(sig)
        self.t += 1
        w = float(self.t)                                    # linear averaging
        for lev in reversed(self.levels):
            self._terminal_and_chance(lev)
            dm = self.kind[lev] == DECISION
            if not dm.any():
                continue
            nodes = lev[dm]
            j = self.dec_id[nodes]
            ch = self.child[j]                                # (n,4)
            Vc = np.where((ch >= 0)[:, :, None], self.V[np.maximum(ch, 0)], 0.0)   # (n,4,S)
            Vc4 = Vc.reshape(-1, 4, K, K)
            s = sig[j]                                        # (n,K,4)
            pl = self.player[nodes]
            for who in (0, 1):
                mm = pl == who
                if not mm.any():
                    continue
                nn, jj, V4, ss = nodes[mm], j[mm], Vc4[mm], s[mm]
                if who == 0:
                    wgt = ss.transpose(0, 2, 1)[:, :, :, None]           # (n,4,K,1)
                    self.V[nn] = (V4 * wgt).sum(axis=1).reshape(-1, S)
                else:
                    wgt = ss.transpose(0, 2, 1)[:, :, None, :]           # (n,4,1,K)
                    self.V[nn] = (V4 * wgt).sum(axis=1).reshape(-1, S)
                if who != update_player:
                    continue
                if who == 0:
                    opp = self.A1[nn].reshape(-1, 1, K, K)
                    cfv = (opp * V4).sum(axis=3)                         # (n,4,K)
                    reach = self.A0[nn].reshape(-1, K, K).sum(axis=2)    # (n,K)
                else:
                    opp = self.A0[nn].reshape(-1, 1, K, K)
                    cfv = -(opp * V4).sum(axis=2)                        # (n,4,K)
                    reach = self.A1[nn].reshape(-1, K, K).sum(axis=1)
                cfv = cfv.transpose(0, 2, 1)                             # (n,K,4)
                base = (cfv * ss).sum(axis=2, keepdims=True)
                r = self.regret[jj] + (cfv - base) * self.legal[jj][:, None, :]
                self.regret[jj] = np.maximum(r, 0)
                self.avg[jj] += w * reach[:, :, None] * ss

    def game_value(self, sig=None) -> float:
        """P0's expected chips per hand when both play sig (default: average)."""
        sig = self.current("avg") if sig is None else sig
        K, S = self.K, self.S
        for lev in reversed(self.levels):
            self._terminal_and_chance(lev)
            dm = self.kind[lev] == DECISION
            if not dm.any():
                continue
            nodes = lev[dm]
            j = self.dec_id[nodes]
            ch = self.child[j]
            Vc4 = np.where((ch >= 0)[:, :, None], self.V[np.maximum(ch, 0)], 0.0).reshape(-1, 4, K, K)
            s = sig[j]
            pl = self.player[nodes]
            out = np.where(pl[:, None, None] == 0,
                           (Vc4 * s.transpose(0, 2, 1)[:, :, :, None]).sum(axis=1),
                           (Vc4 * s.transpose(0, 2, 1)[:, :, None, :]).sum(axis=1))
            self.V[nodes] = out.reshape(-1, S)
        return float(self.init @ self.V[0])

    def best_response_value(self, br: int, sig=None) -> float:
        """Value (chips/hand, from br's view) of the best response of player
        `br` against the other player's strategy sig (default: average)."""
        sig = self.current("avg") if sig is None else sig
        K, S = self.K, self.S
        self._forward(sig)
        for lev in reversed(self.levels):
            self._terminal_and_chance(lev)
            dm = self.kind[lev] == DECISION
            if not dm.any():
                continue
            nodes = lev[dm]
            j = self.dec_id[nodes]
            ch = self.child[j]
            Vc4 = np.where((ch >= 0)[:, :, None], self.V[np.maximum(ch, 0)], 0.0).reshape(-1, 4, K, K)
            pl = self.player[nodes]
            res = np.zeros((len(nodes), K, K), dtype=self.dt)
            for who in (0, 1):
                mm = pl == who
                if not mm.any():
                    continue
                V4, jj, nn = Vc4[mm], j[mm], nodes[mm]
                if who != br:
                    s = sig[jj].transpose(0, 2, 1)
                    res[mm] = (V4 * (s[:, :, :, None] if who == 0 else s[:, :, None, :])).sum(axis=1)
                    continue
                # BR player: pick the action maximizing cf value per own bucket
                if who == 0:
                    opp = self.A1[nn].reshape(-1, 1, K, K)
                    cfv = (opp * V4).sum(axis=3)                     # (n,4,K)
                else:
                    opp = self.A0[nn].reshape(-1, 1, K, K)
                    cfv = -(opp * V4).sum(axis=2)
                cfv = np.where(self.legal[jj][:, :, None], cfv, -np.inf)
                best = cfv.argmax(axis=1)                            # (n,K)
                if getattr(self, "_br_record", None) is not None:
                    self._br_record[jj] = best
                if who == 0:
                    idx = best[:, None, :, None].repeat(K, axis=3)
                    res[mm] = np.take_along_axis(V4, idx, axis=1)[:, 0]
                else:
                    idx = best[:, None, None, :].repeat(K, axis=2)
                    res[mm] = np.take_along_axis(V4, idx, axis=1)[:, 0]
            self.V[nodes] = res.reshape(-1, S)
        v0 = float(self.init @ self.V[0])
        return v0 if br == 0 else -v0

    def best_response_strategy(self, sig=None) -> np.ndarray:
        """A full one-hot strategy: at P0's nodes the best response to sig's
        P1, at P1's nodes the best response to sig's P0 (in the abstraction)."""
        D = len(self.dec)
        out = np.zeros((D, self.K, 4), dtype=np.float32)
        for br in (0, 1):
            self._br_record = np.zeros((D, self.K), dtype=np.int64)
            self.best_response_value(br, sig)
            mine = self.player[self.dec] == br
            idx = self._br_record[mine]
            rows = np.where(mine)[0]
            out[rows[:, None], np.arange(self.K)[None, :], idx] = 1.0
        self._br_record = None
        return out

    def exploitability(self) -> tuple[float, float, float]:
        """(BR vs P1 strategy, BR vs P0 strategy, avg) in bb/hand. If both
        players play our average strategy the sum would be 0 at equilibrium."""
        b0 = self.best_response_value(0)
        b1 = self.best_response_value(1)
        return b0 / 2, b1 / 2, (b0 + b1) / 4

    def save(self, path: str, feature: str = "hs"):
        extra = {"centroids": self.ab["centroids"]} if "centroids" in self.ab else {}
        np.savez_compressed(path, thr=self.ab["thr"], K=self.K, strategy=self.current("avg").astype(np.float32),
                            feature=np.array(feature), **extra,
                            dec_nodes=self.dec, keys=np.array([repr(self.nodes[i].key) for i in self.dec]),
                            players=self.player[self.dec], streets=self.street[self.dec])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["a"])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--buckets", default=None, help="precomputed buckets file (solver/abstraction2.py)")
    ap.add_argument("--feature", default="hs", choices=["hs", "h2", "h2d"],
                    help="h2: data tags are read as deals_<tag>_h2.npz (E[HS^2] buckets)")
    a = ap.parse_args()
    if a.buckets:
        z = np.load(a.buckets)
        t0 = time.time()
        ab = abstraction_from_buckets(z["buckets"], z["outcome"], int(z["K"]), thr=z["thr"], centroids=z["centroids"])
        sv = Solver(ab)
        if a.resume:
            st = np.load(a.resume)
            sv.regret, sv.avg, sv.t = st["regret"], st["avg"], int(st["t"])
        return run(sv, a, t0)
    eqs, ocs = [], []
    for tag in a.data:
        z = np.load(os.path.join(DATA, f"deals_{tag}{'_h2' if a.feature == 'h2' else ''}.npz"))
        eqs.append(z["eq"])
        ocs.append(z["outcome"])
    eq, oc = np.concatenate(eqs), np.concatenate(ocs)
    t0 = time.time()
    ab = build_abstraction(eq, oc, a.K)
    sv = Solver(ab)
    if a.resume:
        z = np.load(a.resume)
        sv.regret, sv.avg, sv.t = z["regret"], z["avg"], int(z["t"])
    print(f"{len(eq)} deals, K={a.K}, {sv.N} nodes, {len(sv.dec)} decisions; setup {time.time() - t0:.1f}s", flush=True)
    return run(sv, a, t0)


def run(sv, a, t0):
    t0 = time.time()
    for it in range(1, a.iters + 1):
        sv.iterate(sv.t % 2)
        if it % a.every == 0 or it == a.iters:
            e0, e1, e = sv.exploitability()
            gv = sv.game_value() / 2
            print(f"iter {sv.t:5d}  {time.time() - t0:6.0f}s  exploitability {e * 1000:7.1f} mbb/hand "
                  f"(BR0 {e0:+.3f}, BR1 {e1:+.3f} bb)  button value {gv:+.3f} bb/hand", flush=True)
    out = a.out or os.path.join(DATA, f"strategy_K{a.K}.npz")
    sv.save(out, a.feature)
    np.savez_compressed(out.replace(".npz", "_state.npz"), regret=sv.regret, avg=sv.avg, t=sv.t)
    print("saved", out)


if __name__ == "__main__":
    main()

"""
lab/leakscan.py - systematic search for spots where OUR bot can be exploited.

Opponent = a strong near-equilibrium base (a previous blueprint) plus ONE
deviation on ONE street, e.g. "on the river, always bet when checked to" or
"on the flop, never fold". If a deviation wins money against our bot, our bot
mishandles that situation - exactly what an adaptive bot would discover.
Each probe is compared with the undeviated base (the noise floor).

    python lab/leakscan.py --seeds 4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import league  # noqa: E402
from evaluate import make  # noqa: E402
from game import Action  # noqa: E402

F, C, RH, RP = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
BASE = os.path.join(os.path.dirname(_HERE), "solver", "data", "strategy_K20h2L.npz")
STREETS = ("pre", "flop", "turn", "river")
DEVIATIONS = ("bet_when_checked", "raise_vs_bet", "never_fold", "fold_vs_bet", "pot_size", "check_only")


class Probe:
    def __init__(self, street: int | None, dev: str | None, prob: float = 1.0, seed: int = 5):
        self.street, self.dev, self.prob, self.seed = street, dev, prob, seed

    def new_half(self, config, m):
        from solverbot import SolverBot
        self.inner = SolverBot(BASE, exploit=False, fold_exploit=False, precision=1)
        self.inner.new_half(config, m)
        m["probe_rng"] = np.random.default_rng(self.seed)

    def on_hand_end(self, result, m):
        self.inner.on_hand_end(result, m)

    def choose_action(self, obs, m):
        a = self.inner.choose_action(obs, m)
        st_ok = self.street == -1 or (isinstance(self.street, tuple) and int(obs.street) in self.street) \
            or int(obs.street) == self.street
        if self.dev is None or not st_ok or m["probe_rng"].random() > self.prob:
            return a
        legal = obs.legal_actions
        can_raise = RH in legal or RP in legal
        facing = obs.to_call > 0
        d = self.dev
        if d == "bet_when_checked" and not facing and can_raise:
            return RH if RH in legal else RP
        if d == "raise_vs_bet" and facing and can_raise:
            return RP if RP in legal else RH
        if d == "never_fold" and a == F:
            return C
        if d == "fold_vs_bet" and facing and F in legal:
            return F
        if d == "pot_size" and a == RH and RP in legal:
            return RP
        if d == "check_only" and a in (RH, RP):
            return C
        return a


def probes():
    out = {"base(no deviation)": lambda: Probe(None, None)}
    for s, sn in enumerate(STREETS):
        for d in DEVIATIONS:
            out[f"{sn}:{d}"] = (lambda s_, d_: (lambda: Probe(s_, d_)))(s, d)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="hybridX@1")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--deals", type=int, default=1000)
    a = ap.parse_args()
    summary, res = league.run({"us": make(a.target)}, probes(), [f"leak{i}" for i in range(a.seeds)],
                              a.deals, verbose=False)
    per = summary[0][4]
    base = per["base(no deviation)"][1]
    print(f"our bb/deal vs the undeviated base: {base:+.2f} (noise floor)\n")
    print(f"{'probe':30s} {'our bb/deal':>11s} {'vs base':>8s}  {'points':>6s}")
    for name, (pt, bb) in sorted(per.items(), key=lambda kv: kv[1][1]):
        flag = "  <-- LEAK?" if bb < base - 0.5 or pt < 0.5 else ""
        print(f"{name:30s} {bb:+11.2f} {bb - base:+8.2f}  {pt:6.2f}{flag}")

"""
lab/gauntlet.py - every opponent from the whole project, in one run.

    python lab/gauntlet.py --cands hybridX@1 "hybridX@1:..." --seeds 2

Opponents: the full pool (simple, AI-typical, official-evaluator, weird),
closest red-team attackers from every round, rule-targeting bots (rare
bluffers, style switchers, fold-equity hunters, stab-everything), skewed and
evolved solvers, the white-box best response, drew stand-ins, hero, previous
solvers, and the teammate's bot. Ranked by worst case, then points, then bb.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import league  # noqa: E402
import pool as poolmod  # noqa: E402
import tune_x  # noqa: E402
from evaluate import make  # noqa: E402
from leakscan import Probe  # noqa: E402
from sim import load_agent_file  # noqa: E402

SD = os.path.join(os.path.dirname(_HERE), "solver", "data")


def opponents():
    o = dict(poolmod.POOL)
    o.update(tune_x.guards())
    K20cd = os.path.join(SD, "strategy_K20cd.npz")
    L = os.path.join(SD, "strategy_K20h2L.npz")
    o["drew_extreme"] = lambda: poolmod.DrewProxy(K20cd)
    o["drew_matched"] = lambda: poolmod.DrewProxy(K20cd, seed=62, fourbet_eq=0.53, stab=0.25, river_call_eq=0.33)
    o["rare_bluff_hi"] = lambda: poolmod.RareBluffer(0.35, 22)
    o["switch200"] = lambda: poolmod.Switcher(200, 41)
    o["switch500"] = lambda: poolmod.Switcher(500, 44)
    o["hunter"] = lambda: poolmod.FoldEquityHunter(L)
    o["hunter_aggr"] = lambda: poolmod.FoldEquityHunter(L, margin=0.0, prior_w=4, seed=52)
    o["stab_all"] = lambda: Probe(-1, "bet_when_checked")
    o["evolved_passive"] = lambda: poolmod.SkewedSolver(K20cd, aggro=0.0, passive=0.21, seed=7)
    o["evolved_mix"] = lambda: poolmod.SkewedSolver(K20cd, aggro=0.07, passive=0.23, seed=8)
    br = os.path.join(SD, "br_vs_P.npz")
    if os.path.exists(br):
        o["whitebox_br"] = make(f"solver:{br}@1")
    o["solver_K20cd"] = make(f"solver:{K20cd}@1")
    tm = os.path.join(os.path.dirname(_HERE), "opponents", "teammate_v5", "agent.py")
    if os.path.exists(tm):
        o["teammate_v5"] = load_agent_file(tm)
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--tag", default="gx")
    a = ap.parse_args()
    opps = opponents()
    cands = {c: make(c) for c in a.cands}
    summary, _ = league.run(cands, opps, [f"{a.tag}{i}" for i in range(a.seeds)], a.deals,
                            procs=a.procs, verbose=False)
    for c, worst, total, mbb, per in summary:
        losses = [k for k, v in per.items() if v[0] < 0.5]
        draws = [k for k, v in per.items() if v[0] == 0.5]
        print(f"\n== {c}: worst {worst:.2f}  points {total:.1f}/{len(opps)}  mean {mbb:+.2f} bb/deal")
        print(f"   below 0.5 points: {losses or 'none'}")
        print(f"   draws: {draws}")
        tight = sorted(per.items(), key=lambda kv: kv[1][1])[:10]
        print("   closest 10:", ", ".join(f"{k} {v[1]:+.2f}" for k, v in tight))

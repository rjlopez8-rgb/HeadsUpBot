"""
lab/evaluate.py - head-to-head evaluation of bot versions against the pool
and against each other (round robin), ranked by worst-case points.

    python lab/evaluate.py --cands hero solver:data/strategy.npz solverX:data/strategy.npz \
                           --deals 1000 --seeds 2 [--opps strong|all|none] [--extra path/agent.py ...]

Candidate syntax:
    hero                    hero.py with default Params
    hero:<champion.json>    hero.py with tuned Params
    solver:<npz>            blueprint only (no exploit layer)
    solverX:<npz>           blueprint + exploit layer
    hybrid:<npz>            blueprint, switching to hero-style exploitation vs non-equilibrium opponents
    file:<agent.py>         any agent.py-style file
Every candidate also plays every other candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import league  # noqa: E402
import pool as poolmod  # noqa: E402
from sim import load_agent_file  # noqa: E402

STRONG = ["eq_basic", "eq_tight", "eq_loose_aggr", "eq_modeling", "eq_slow", "exploiter",
          "guide_example", "official_mc", "official_mc_fast",
          "check_raiser", "river_bluffer", "pf_shover", "maniac", "station", "checkfold"]


def make(spec: str):
    from hero import Hero, Params
    kind, _, arg = spec.partition(":")
    if kind == "hero":
        if arg:
            d = json.load(open(arg))
            d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
            return lambda: Hero(Params(**d))
        return lambda: Hero(Params())
    if kind in ("solver", "solverX"):
        from solverbot import SolverBot
        path = arg or os.path.join(os.path.dirname(_HERE), "data", "strategy.npz")
        ex = kind == "solverX"
        return lambda: SolverBot(path, exploit=ex)
    if kind == "hybrid":
        from solverbot import Hybrid
        path = arg or os.path.join(os.path.dirname(_HERE), "data", "strategy.npz")
        return lambda: Hybrid(path)
    if kind == "file":
        return load_agent_file(arg)
    raise ValueError(spec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--opps", default="strong")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    cands = {c: make(c) for c in a.cands}
    opps = {}
    if a.opps == "all":
        opps.update(poolmod.POOL)
    elif a.opps == "strong":
        opps.update({k: poolmod.POOL[k] for k in STRONG})
    for path in a.extra:
        f = load_agent_file(path)
        opps["x:" + f.__name__] = f
    for c in a.cands:                     # round robin between candidates
        opps["vs:" + c] = cands[c]
    league.run(cands, opps, [f"ev{i}" for i in range(a.seeds)], a.deals)

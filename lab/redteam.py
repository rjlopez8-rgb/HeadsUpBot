"""
lab/redteam.py - try hard to find ANY bot that beats ours.

Generates many attacker variants (heavily mutated hero bots, exploiters,
equity bots with extreme settings), plays each against our bot in full
duplicate matches, and reports every attacker that wins or comes close.

    python lab/redteam.py --target hybridX:path.npz --n 30 --deals 1000 --seeds 2
"""
from __future__ import annotations

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import league  # noqa: E402
import pool as poolmod  # noqa: E402
from evaluate import make  # noqa: E402
from hero import Hero, Params  # noqa: E402
from tune import mutate  # noqa: E402


def attackers(n: int, seed: int):
    rng = random.Random(seed)
    out = {}
    for i in range(n):
        kind = i % 3
        if kind == 0:                       # mutated hero (adaptive, range-reading)
            p = mutate(Params(), rng, strength=2.5)
            p.seed = rng.randint(0, 10**6)
            out[f"hero_m{i}"] = (lambda q: (lambda: Hero(q)))(p)
        elif kind == 1:                     # extreme equity bot
            kw = dict(raise_eq=rng.uniform(0.45, 0.8), pot_eq=rng.uniform(0.6, 0.9),
                      margin=rng.uniform(-0.15, 0.1), bet_eq=rng.uniform(0.35, 0.75),
                      bluff=rng.uniform(0, 0.4), seed=rng.randint(0, 999), n=150)
            out[f"eq_x{i}"] = (lambda k: (lambda: poolmod.EquityBot(**k)))(kw)
        else:                               # hyper-aggressive mutated hero
            p = mutate(Params(), rng, strength=2.0)
            p.bet_t = tuple(max(0.3, x - 0.12) for x in p.bet_t)
            p.raise_t = tuple(max(0.35, x - 0.1) for x in p.raise_t)
            p.bluff_base = min(0.4, p.bluff_base + 0.15)
            p.seed = rng.randint(0, 10**6)
            out[f"aggro_m{i}"] = (lambda q: (lambda: Hero(q)))(p)
    out["exploiter"] = poolmod.POOL["exploiter"]
    out["eq_modeling"] = poolmod.POOL["eq_modeling"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="hybridX")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--rseed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args()
    atk = attackers(a.n, a.rseed)
    summary, res = league.run({"target": make(a.target)}, atk, [f"rt{a.rseed}_{i}" for i in range(a.seeds)],
                              a.deals, procs=a.procs, verbose=False)
    _, worst, total, mbb, per = summary[0]
    rows = sorted(per.items(), key=lambda kv: kv[1][1])
    print(f"target {a.target}: {total:.1f}/{len(atk)} points, worst {worst:.2f}, mean {mbb:+.2f} bb/deal")
    print("closest attackers (target's points, target's bb/deal):")
    for name, (pt, bb) in rows[:8]:
        print(f"   {name:12s} {pt:.2f}  {bb:+.2f}")
    beaten = [n for n, (pt, bb) in per.items() if pt < 0.5]
    print("ATTACKERS THAT BEAT US:", beaten or "none")

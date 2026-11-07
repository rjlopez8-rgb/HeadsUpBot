"""
lab/tune.py - evolve Hero's Params: mutate the champion, play every mutant
against the strong part of the pool AND against the champion itself (plus
earlier champions), keep the best by worst-case points, then points, then bb.

    python lab/tune.py --cands 10 --deals 600 --seeds 2
Results append to lab/results/tune_log.jsonl; the champion is saved to
lab/results/champion.json (load it with hero.Params(**json)).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict, fields

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import league  # noqa: E402
import pool as poolmod  # noqa: E402
from hero import Hero, Params  # noqa: E402

RES = os.path.join(_HERE, "results")
STRONG = ["eq_basic", "eq_tight", "eq_loose_aggr", "eq_modeling", "exploiter",
          "check_raiser", "river_bluffer", "pf_shover", "maniac", "station"]

# (field, kind, lo, hi, step-scale)
SPACE = [
    ("call_margin", "f", -0.06, 0.08, 0.02),
    ("range_k", "f", 0.8, 3.0, 0.3),
    ("range_min", "f", 0.05, 0.5, 0.06),
    ("reraise_shrink", "f", 0.4, 1.0, 0.1),
    ("bluff_base", "f", 0.0, 0.25, 0.04),
    ("bluff_adapt", "f", 0.0, 2.0, 0.3),
    ("bluff_floor", "f", 0.0, 0.5, 0.06),
    ("thin_value", "f", 0.0, 0.6, 0.08),
    ("size_mix", "f", 0.0, 0.5, 0.08),
    ("var_ctrl", "f", -0.1, 0.1, 0.03),
    ("bet_t", "t", 0.45, 0.80, 0.03),
    ("raise_t", "t", 0.50, 0.85, 0.03),
    ("pot_t", "t", 0.60, 0.92, 0.03),
]


def mutate(p: Params, rng: random.Random, strength: float = 1.0) -> Params:
    q = copy.deepcopy(p)
    chosen = rng.sample(SPACE, k=rng.randint(2, 5))
    for name, kind, lo, hi, sc in chosen:
        if kind == "f":
            v = getattr(q, name) + rng.gauss(0, sc * strength)
            setattr(q, name, round(min(hi, max(lo, v)), 4))
        else:
            shift = rng.gauss(0, sc * strength)
            t = tuple(round(min(hi, max(lo, x + shift + rng.gauss(0, sc * 0.3))), 4) for x in getattr(q, name))
            setattr(q, name, t)
    return q


def load_champion() -> Params:
    path = os.path.join(RES, "champion.json")
    if os.path.exists(path):
        d = json.load(open(path))
        for f in fields(Params):
            if isinstance(getattr(Params(), f.name), tuple) and f.name in d:
                d[f.name] = tuple(d[f.name])
        return Params(**{k: v for k, v in d.items() if k in {f.name for f in fields(Params)}})
    return Params()


def factory(p: Params):
    return lambda: Hero(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", type=int, default=10)
    ap.add_argument("--deals", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--rseed", type=int, default=int(time.time()))
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    rng = random.Random(a.rseed)
    champ = load_champion()
    cands = {"champion": factory(champ)}
    params = {"champion": champ}
    for i in range(a.cands):
        q = mutate(champ, rng, a.strength)
        cands[f"m{i}"] = factory(q)
        params[f"m{i}"] = q
    opps = {k: poolmod.POOL[k] for k in STRONG}
    opps["CHAMPION"] = factory(champ)          # every mutant must also beat the champion
    seeds = [f"tune{a.rseed}_{i}" for i in range(a.seeds)]
    summary, res = league.run(cands, opps, seeds, a.deals, verbose=False)
    # champion vs itself is exactly 0 by symmetry: count it as a chop
    with open(os.path.join(RES, "tune_log.jsonl"), "a") as fh:
        for c, worst, total, mbb, per in summary:
            fh.write(json.dumps({"t": time.time(), "name": c, "worst": worst, "total": total,
                                 "mean_bb": mbb, "vs_champ_bb": per["CHAMPION"][1],
                                 "params": asdict(params[c])}) + "\n")
    print(f"{'name':10s} {'worst':>5s} {'pts':>5s} {'meanbb':>7s} {'vsChamp':>8s}")
    for c, worst, total, mbb, per in summary:
        print(f"{c:10s} {worst:5.2f} {total:5.1f} {mbb:+7.2f} {per['CHAMPION'][1]:+8.3f}")
    best = summary[0]
    # promote only if it does not lose to the champion head-to-head
    # Noise check: the champion vs an identical copy of itself is pure luck, so a
    # mutant must beat the champion head-to-head by more than MIN_EDGE bb/deal
    # AND not score fewer points vs the pool. Re-test promising mutants with
    # more seeds/deals before trusting them.
    MIN_EDGE = 0.25
    champ_row = next(r for r in summary if r[0] == "champion")
    if (best[0] != "champion" and best[4]["CHAMPION"][1] > MIN_EDGE
            and best[2] >= champ_row[2] and best[3] >= champ_row[3]):
        json.dump(asdict(params[best[0]]), open(os.path.join(RES, "champion.json"), "w"), indent=1)
        print("NEW CHAMPION:", best[0], asdict(params[best[0]]))
    else:
        print("champion kept")

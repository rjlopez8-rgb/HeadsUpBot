"""
lab/tune_x.py - tune HybridX's exploit settings (HParams + OParams) safely.

Each candidate plays two opponent sets on common seeds:
  TARGETS - exploitable bots: we want BIGGER margins against them
  GUARDS  - the hardest opponents found so far (closest red-team attackers,
            rule-targeting bluffers, skewed solver, hero, previous solver)
A candidate is REJECTED if it loses (or trails by > GUARD_TOL bb/deal) to any
guard. Among the rest: score = mean target bb + 0.5 * mean guard bb.

    python lab/tune_x.py --cands 12 --rounds 1 --seeds 1
Champion settings -> lab/results/x_champion.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import league  # noqa: E402
import pool as poolmod  # noqa: E402
from hero import Hero, Params  # noqa: E402
from redteam import attackers  # noqa: E402
from solverbot import HParams, HybridX, OParams  # noqa: E402

RES = os.path.join(_HERE, "results")
GUARD_TOL = -0.3
PREC = float(os.environ.get("TUNE_PRECISION", "1"))      # speed: tuning at precision 1
SOLVER_DIR = os.path.join(os.path.dirname(_HERE), "solver", "data")

TARGETS = ["eq_basic", "eq_tight", "eq_loose_aggr", "eq_modeling", "official_mc_fast",
           "guide_example", "check_raiser", "river_bluffer", "station", "maniac"]

SPACE_O = [("margin", -0.02, 0.10, 0.02), ("range_k", 1.0, 2.6, 0.25), ("range_min", 0.06, 0.35, 0.05),
           ("tight_ratio", 0.5, 1.0, 0.08), ("fold_gap", 0.04, 0.2, 0.03), ("weak_frac", 0.25, 0.6, 0.06),
           ("value_eq", 0.5, 0.75, 0.03), ("bluff_scale", 0.4, 2.5, 0.3)]
SPACE_H = [("decide_at", 60, 200, 25), ("d_hi", 0.09, 0.2, 0.02), ("d_lo", 0.04, 0.12, 0.015)]


def guards():
    g = {}
    a1, a2, a3 = attackers(18, 1), attackers(18, 2), attackers(24, 3)
    g["r1_aggro14"], g["r1_aggro11"] = a1["aggro_m14"], a1["aggro_m11"]
    g["r2_aggro17"], g["r2_hero0"] = a2["aggro_m17"], a2["hero_m0"]
    g["r3_hero9"], g["r3_aggro2"] = a3["hero_m9"], a3["aggro_m2"]
    g["rare_bluff"] = lambda: poolmod.RareBluffer()
    g["skew_aggro40"] = lambda: poolmod.SkewedSolver(os.path.join(SOLVER_DIR, "strategy_K20cd.npz"), aggro=0.4, seed=32)
    g["hero"] = lambda: Hero(Params())
    from solverbot import SolverBot
    g["solver_prev"] = lambda: SolverBot(os.path.join(SOLVER_DIR, "strategy_K20h2L.npz"), exploit=False,
                                         precision=PREC)
    return g


def mutate(hp: HParams, op: OParams, rng: random.Random):
    hp, op = copy.deepcopy(hp), copy.deepcopy(op)
    for name, lo, hi, sc in rng.sample(SPACE_O, k=rng.randint(1, 3)):
        setattr(op, name, round(min(hi, max(lo, getattr(op, name) + rng.gauss(0, sc))), 4))
    if rng.random() < 0.5:
        name, lo, hi, sc = rng.choice(SPACE_H)
        v = min(hi, max(lo, getattr(hp, name) + rng.gauss(0, sc)))
        setattr(hp, name, int(v) if name == "decide_at" else round(v, 4))
    if hp.d_lo >= hp.d_hi:
        hp.d_lo = round(hp.d_hi * 0.6, 4)
    return hp, op


def load_champion():
    path = os.path.join(RES, "x_champion.json")
    if os.path.exists(path):
        d = json.load(open(path))
        return HParams(**d["hp"]), OParams(**d["op"])
    return HParams(), OParams()


def factory(hp, op):
    return lambda: HybridX(hp=hp, op=op, precision=PREC)


def score_rows(per, gnames, tnames):
    g = [per[n][1] for n in gnames]
    t = [per[n][1] for n in tnames]
    # solver-vs-solver single matches swing ~±0.4 bb/deal on their own (strategy
    # randomness), so the solver guard only rejects on a clearly large deficit
    lost = [n for n in gnames if (per[n][1] < -0.8) if n == "solver_prev"] + \
           [n for n in gnames if n != "solver_prev" and (per[n][0] < 0.5 or per[n][1] < GUARD_TOL)]
    return sum(t) / len(t) + 0.5 * sum(g) / len(g), sum(t) / len(t), sum(g) / len(g), lost


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--rseed", type=int, default=int(time.time()) % 100000)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    rng = random.Random(a.rseed)
    G = guards()
    opps = {**{k: poolmod.POOL[k] for k in TARGETS}, **G}
    for rnd in range(a.rounds):
        hp0, op0 = load_champion()
        cands, conf = {"champion": factory(hp0, op0)}, {"champion": (hp0, op0)}
        for i in range(a.cands):
            hp, op = mutate(hp0, op0, rng)
            cands[f"c{i}"], conf[f"c{i}"] = factory(hp, op), (hp, op)
        seeds = [f"tx{a.rseed}_{rnd}_{i}" for i in range(a.seeds)]
        t0 = time.time()
        summary, _ = league.run(cands, opps, seeds, a.deals, verbose=False)
        rows = []
        for c, worst, total, mbb, per in summary:
            sc, t, g, lost = score_rows(per, list(G), TARGETS)
            rows.append((sc, c, t, g, lost))
        rows.sort(reverse=True)
        print(f"round {rnd}: {len(cands)} candidates in {time.time() - t0:.0f}s")
        for sc, c, t, g, lost in rows:
            print(f"   {c:9s} score {sc:+.3f}  targets {t:+.2f}  guards {g:+.2f}  {'REJECT ' + str(lost) if lost else ''}")
        champ = next(r for r in rows if r[1] == "champion")
        ok = [r for r in rows if not r[4]]
        with open(os.path.join(RES, "x_log.jsonl"), "a") as fh:
            for sc, c, t, g, lost in rows:
                hp, op = conf[c]
                fh.write(json.dumps({"round": rnd, "name": c, "score": sc, "targets": t, "guards": g,
                                     "lost": lost, "hp": asdict(hp), "op": asdict(op)}) + "\n")
        if ok and ok[0][1] != "champion" and ok[0][0] > champ[0] + 0.15:
            hp, op = conf[ok[0][1]]
            json.dump({"hp": asdict(hp), "op": asdict(op)}, open(os.path.join(RES, "x_champion.json"), "w"), indent=1)
            print("   NEW CHAMPION", ok[0][1], asdict(hp), asdict(op), flush=True)
        else:
            print("   champion kept", flush=True)

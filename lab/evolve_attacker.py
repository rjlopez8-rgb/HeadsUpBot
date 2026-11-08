"""
lab/evolve_attacker.py - evolve opponents that are optimized to beat OUR bot.

A population of attackers from three families (tuned heuristic "hero" bots,
skewed solvers, adaptive fold-equity hunters) plays our bot on common seeds;
each generation keeps the best and breeds mutated children. Fitness = the
attacker's bb/deal against us. The final champion is re-tested on fresh
seeds (best-of-many on few seeds is always optimistic).

    python lab/evolve_attacker.py --gens 20 --pop 12 --seeds 2
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import league  # noqa: E402
import pool as poolmod  # noqa: E402
from evaluate import make  # noqa: E402
from hero import Hero, Params  # noqa: E402
from tune import mutate as mutate_hero  # noqa: E402

SD = os.path.join(os.path.dirname(_HERE), "solver", "data")
BASES = ["strategy_K20cd.npz", "strategy_K20h2L.npz"]


def build(g):
    fam = g["fam"]
    if fam == "hero":
        p = g["p"]
        return lambda: Hero(p)
    if fam == "skew":
        return lambda: poolmod.SkewedSolver(os.path.join(SD, g["base"]), aggro=g["aggro"],
                                            passive=g["passive"], seed=g["seed"])
    return lambda: poolmod.FoldEquityHunter(os.path.join(SD, g["base"]), margin=g["margin"],
                                            prior_w=g["prior_w"], seed=g["seed"], value_read=g["vr"])


def random_genome(rng):
    fam = rng.choice(["hero", "hero", "skew", "feh"])
    if fam == "hero":
        p = mutate_hero(Params(), rng, strength=2.0)
        p.seed = rng.randint(0, 10**6)
        return {"fam": fam, "p": p}
    if fam == "skew":
        return {"fam": fam, "base": rng.choice(BASES), "aggro": rng.uniform(0, 0.5),
                "passive": rng.uniform(0, 0.5), "seed": rng.randint(0, 999)}
    return {"fam": fam, "base": rng.choice(BASES), "margin": rng.uniform(-0.05, 0.1),
            "prior_w": rng.uniform(2, 15), "seed": rng.randint(0, 999), "vr": rng.random() < 0.7}


def child(g, rng):
    g = copy.deepcopy(g)
    if g["fam"] == "hero":
        g["p"] = mutate_hero(g["p"], rng, strength=0.8)
        g["p"].seed = rng.randint(0, 10**6)
    elif g["fam"] == "skew":
        g["aggro"] = min(0.8, max(0, g["aggro"] + rng.gauss(0, 0.08)))
        g["passive"] = min(0.8, max(0, g["passive"] + rng.gauss(0, 0.08)))
        g["seed"] = rng.randint(0, 999)
    else:
        g["margin"] = g["margin"] + rng.gauss(0, 0.03)
        g["prior_w"] = min(30, max(1, g["prior_w"] + rng.gauss(0, 3)))
        g["seed"] = rng.randint(0, 999)
    return g


def describe(g):
    if g["fam"] == "hero":
        p = g["p"]
        return (f"hero bet_t={tuple(round(x, 2) for x in p.bet_t)} raise_t={tuple(round(x, 2) for x in p.raise_t)} "
                f"call_margin={p.call_margin:.2f} bluff={p.bluff_base:.2f}/{p.bluff_adapt:.2f}")
    if g["fam"] == "skew":
        return f"skew {g['base']} aggro={g['aggro']:.2f} passive={g['passive']:.2f}"
    return f"hunter {g['base']} margin={g['margin']:.3f} prior_w={g['prior_w']:.1f} value_read={g['vr']}"


def evaluate(target, genomes, seeds, deals):
    opps = {f"g{i}": build(g) for i, g in enumerate(genomes)}
    summary, _ = league.run({"us": target}, opps, seeds, deals, verbose=False)
    per = summary[0][4]
    return [(-per[f"g{i}"][1], 1 - per[f"g{i}"][0]) for i in range(len(genomes))]   # attacker bb, attacker pts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="hybridX@1")
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--rseed", type=int, default=7)
    ap.add_argument("--validate", type=int, default=8)
    a = ap.parse_args()
    rng = random.Random(a.rseed)
    target = make(a.target)
    popn = [random_genome(rng) for _ in range(a.pop)]
    best_ever = None
    for gen in range(a.gens):
        t0 = time.time()
        seeds = [f"evo{a.rseed}_{gen}_{i}" for i in range(a.seeds)]      # fresh seeds each generation
        fit = evaluate(target, popn, seeds, a.deals)
        ranked = sorted(zip(fit, range(len(popn))), key=lambda x: x[0][0], reverse=True)
        top = [popn[i] for _, i in ranked[: max(3, a.pop // 4)]]
        (bb, pts), i0 = ranked[0]
        print(f"gen {gen:2d} ({time.time() - t0:.0f}s): best attacker {bb:+.2f} bb/deal vs us "
              f"(median {sorted(f[0] for f in fit)[len(fit) // 2]:+.2f}) :: {describe(popn[i0])}", flush=True)
        if best_ever is None or bb > best_ever[0]:
            best_ever = (bb, popn[i0])
        popn = top + [child(rng.choice(top), rng) for _ in range(a.pop - len(top) - 2)] + \
            [random_genome(rng) for _ in range(2)]
    # validate the final elite on fresh seeds
    finals = top[:3]
    vseeds = [f"evoval{a.rseed}_{i}" for i in range(a.validate)]
    vfit = evaluate(target, finals, vseeds, a.deals)
    print("\nVALIDATION on fresh seeds (attacker's bb/deal vs us, attacker's points per match):")
    for (bb, pts), g in zip(vfit, finals):
        print(f"   {bb:+.2f} bb/deal, {pts:.2f} pts :: {describe(g)}")
    beat = [g for (bb, pts), g in zip(vfit, finals) if pts > 0.5]
    print("EVOLVED ATTACKERS THAT BEAT US:", [describe(g) for g in beat] or "none")

"""
lab/league.py - play candidates against the opponent pool and rank them.

    python lab/league.py                      # Hero (default params) vs whole pool
    python lab/league.py --deals 400 --seeds 2
    python lab/league.py --extra path/to/teammate/agent.py   # add any agent.py to the pool

Ranking rule (protects against getting exploited): candidates are ranked by
  1) WORST-case points over opponents, then 2) total points, then 3) mean bb/deal.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import pool as poolmod  # noqa: E402
from sim import load_agent_file, play_match  # noqa: E402

_CANDS: dict = {}
_OPPS: dict = {}


def _job(args):
    cname, oname, seed, deals = args
    m = play_match(_CANDS[cname], _OPPS[oname], seed, deals)
    return cname, oname, seed, m.points1, m.mean_bb, m.verdict, m.faults()


def run(cands: dict, opps: dict, seeds, deals: int, procs: int | None = None, verbose=True):
    global _CANDS, _OPPS
    _CANDS, _OPPS = cands, opps
    jobs = [(c, o, s, deals) for c in cands for o in opps for s in seeds]
    t0 = time.time()
    procs = procs or os.cpu_count() or 1
    if procs > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            rows = pool.map(_job, jobs, chunksize=1)
    else:
        rows = [_job(j) for j in jobs]
    res = {c: {o: [] for o in opps} for c in cands}
    for c, o, s, pts, bb, v, faults in rows:
        res[c][o].append((pts, bb, v, faults))
    summary = []
    for c in cands:
        per = {o: (sum(r[0] for r in res[c][o]) / len(seeds), sum(r[1] for r in res[c][o]) / len(seeds))
               for o in opps}
        worst = min(v[0] for v in per.values())
        total = sum(v[0] for v in per.values())
        mean_bb = sum(v[1] for v in per.values()) / len(per)
        summary.append((c, worst, total, mean_bb, per))
    summary.sort(key=lambda r: (r[1], r[2], r[3]), reverse=True)
    if verbose:
        print(f"{len(jobs)} matches x {deals} deals in {time.time() - t0:.0f}s")
        for c, worst, total, mbb, per in summary:
            print(f"\n== {c}: worst={worst:.2f} points={total:.1f}/{len(opps)} mean={mbb:+.2f} bb/deal")
            for o, (pt, bb) in per.items():
                tag = "WIN " if pt == 1 else "LOSS" if pt == 0 else "chop" if pt == 0.5 else f"{pt:.2f}"
                print(f"   {o:16s} {tag}  {bb:+7.2f} bb/deal")
    return summary, res


if __name__ == "__main__":
    from hero import Hero, Params
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--faults", action="store_true")
    a = ap.parse_args()
    opps = dict(poolmod.POOL)
    for path in a.extra:
        f = load_agent_file(path)
        opps[f.__name__] = f
    if a.only:
        opps = {k: v for k, v in opps.items() if k in a.only}
    summ, res = run({"hero": lambda: Hero(Params())}, opps, [f"s{i}" for i in range(a.seeds)], a.deals)
    if a.faults:
        for o, rows in res["hero"].items():
            print(f"\n-- {o}\n{rows[0][3]}")

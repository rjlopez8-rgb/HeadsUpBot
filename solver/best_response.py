"""
solver/best_response.py - build the white-box attacker: the exact best
response (in the abstraction) to a solved blueprint, saved in the same
format so it plays through solverbot.SolverBot.

    python solver/best_response.py --data c d e f g --feature h2 --K 20 \
        --state solver/data/strategy_K20h2P_state.npz --out solver/data/br_vs_P.npz
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from cfr import DATA, Solver, build_abstraction  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--feature", default="h2")
ap.add_argument("--K", type=int, default=20)
ap.add_argument("--state", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
eqs, ocs = [], []
for tag in a.data:
    z = np.load(os.path.join(DATA, f"deals_{tag}{'_h2' if a.feature == 'h2' else ''}.npz"))
    eqs.append(z["eq"]); ocs.append(z["outcome"])
ab = build_abstraction(np.concatenate(eqs), np.concatenate(ocs), a.K)
sv = Solver(ab)
st = np.load(a.state)
sv.regret, sv.avg = st["regret"], st["avg"]
br = sv.best_response_strategy()
v0, v1 = sv.best_response_value(0), sv.best_response_value(1)
print(f"BR value vs blueprint (abstraction): as button {v0 / 2:+.3f} bb/hand, as BB {v1 / 2:+.3f} bb/hand; "
      f"exploitability {(v0 + v1) / 4 * 1000:.1f} mbb/hand")
np.savez_compressed(a.out, thr=ab["thr"], K=a.K, strategy=br, feature=np.array(a.feature),
                    dec_nodes=sv.dec, keys=np.array([repr(sv.nodes[i].key) for i in sv.dec]),
                    players=sv.player[sv.dec], streets=sv.street[sv.dec])
print("saved", a.out)

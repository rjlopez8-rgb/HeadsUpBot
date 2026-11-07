"""Re-derive the preflop feature of saved deals from data/preflop_eq.npy
(run after rebuilding the table). Usage: python solver/refresh_preflop.py c d e f g"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pokerlib as pl  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
for tag in sys.argv[1:]:
    for suffix in ("", "_h2"):
        path = os.path.join(DATA, f"deals_{tag}{suffix}.npz")
        if not os.path.exists(path):
            continue
        z = dict(np.load(path))
        cards = z["cards"].astype(np.int64)
        z["eq"][:, 0, 0] = pl.preflop_eq(cards[:, 0:5])
        z["eq"][:, 1, 0] = pl.preflop_eq(cards[:, 5:10])
        np.savez_compressed(path, **z)
        print("refreshed", os.path.basename(path), flush=True)

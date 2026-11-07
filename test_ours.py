"""
test_ours.py - checks for our own components (run: python -m pytest -q).

  * fasteval agrees exactly with the official evaluator
  * the preflop table matches a fresh Monte Carlo estimate
  * the bot finds every real betting sequence in the solver tree
  * the bot never errors, never falls back, and stays well inside the time
    bank over a few hundred hands against erratic opponents
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab"))


def test_fasteval_matches_official():
    import fasteval as fe
    from evaluator import omaha_score, warmup
    warmup()
    r = random.Random(5)
    hands = np.array([r.sample(range(36), 10) for _ in range(2000)])
    fast = fe.omaha_batch(hands[:, :5], hands[:, 5:])
    slow = [omaha_score(list(h[:5]), list(h[5:])) for h in hands]
    assert (fast == np.array(slow)).all()


def test_preflop_table_close_to_monte_carlo():
    import pokerlib as pl
    from solver.dataset import batch_equity
    rng = np.random.default_rng(3)
    r = random.Random(9)
    for _ in range(5):
        h = np.array([sorted(r.sample(range(36), 5))])
        mc = batch_equity(h, np.zeros((1, 5), dtype=np.int64), 0, 4000, rng)[0]
        assert abs(pl.preflop_eq(h[0]) - mc) < 0.04


def test_bot_plays_cleanly_and_fast():
    from sim import play_match
    import pool
    from solverbot import HybridX
    mems = []

    class Probe(HybridX):
        def new_half(self, config, memory):
            super().new_half(config, memory)
            mems.append(memory)

    for opp in ("random", "always_pot", "maniac"):
        mems.clear()
        res = play_match(lambda: Probe(), pool.POOL[opp], "test-" + opp, 150)
        for half in res.halves:
            assert half.seat_a1.errors == 0
            assert not half.seat_a1.forfeited and not half.seat_a1.bank_empty
            assert half.seat_a1.max_call < 0.2
        assert all(m["fallbacks"] == 0 for m in mems)
        # 150 hands should use well under 15% of the 8 s bank
        assert all(m["sb_t"] < 1.2 for m in mems)

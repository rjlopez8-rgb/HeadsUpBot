"""
lab/sim.py - fast in-process duplicate matches (no HTTP).

Plays exactly like judge_engine.play_half / combine_halves:
  * same dealing (game.make_deal), same buttons, same Game engine
  * half 2 replays the same deals with seats swapped
  * every half uses a FRESH bot instance and a fresh memory dict
  * errors (exception / illegal action) -> default action; 5 errors = forfeit
  * time bank: each call is timed (minus the 1 ms grace); an empty bank means
    default actions (check/fold) for the rest of the half; a call > 1 s = error
  * verdict: 95% CI on per-deal paired bb; forfeit of either half = loss

A "bot spec" is a zero-argument factory returning an object with
new_half(config, memory), choose_action(obs, memory), on_hand_end(result, memory).
Use load_agent_file(path) to wrap any agent.py-style file (yours, a
teammate's, a baseline) as a spec.

TIME_SCALE multiplies measured thinking time, to pretend the tournament
machine is slower than this one (default 2.0 = assume half speed).
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from game import (ACTION_TIMEOUT_SECONDS, BIG_BLIND, CHOP_Z, ERROR_LIMIT,  # noqa: E402
                  NETWORK_GRACE_SECONDS, TIME_BANK_SECONDS, AgentConfig, Action,
                  Game, button_for_deal, default_action, make_deal)

TIME_SCALE = float(os.environ.get("SIM_TIME_SCALE", "2.0"))


@dataclass
class Seat:
    bot: Any
    memory: dict = field(default_factory=dict)
    time_used: float = 0.0
    calls: int = 0
    errors: int = 0
    forfeited: bool = False
    bank_empty: bool = False
    first_error: str = ""
    max_call: float = 0.0

    def _err(self, msg: str) -> None:
        self.errors += 1
        if not self.first_error:
            self.first_error = msg
        if self.errors >= ERROR_LIMIT:
            self.forfeited = True

    def _timed(self, fn, *args):
        t0 = time.perf_counter()
        try:
            out = fn(*args)
            exc = None
        except Exception as e:  # noqa: BLE001
            out, exc = None, e
        dt = (time.perf_counter() - t0) * TIME_SCALE
        self.calls += 1
        self.max_call = max(self.max_call, dt)
        self.time_used += max(0.0, dt - NETWORK_GRACE_SECONDS)
        if self.time_used >= TIME_BANK_SECONDS:
            self.bank_empty = True
        if dt > ACTION_TIMEOUT_SECONDS:
            self._err(f"call took {dt:.2f}s (scaled)")
            return None, True
        if exc is not None:
            self._err(f"{type(exc).__name__}: {exc}")
            return None, True
        return out, False

    def start(self, config: AgentConfig) -> None:
        _, bad = self._timed(self.bot.new_half, config, self.memory)
        if bad:
            self.forfeited = True

    def act(self, obs) -> Action:
        fallback = default_action(obs.to_call)
        if self.forfeited or self.bank_empty:
            return fallback
        out, bad = self._timed(self.bot.choose_action, obs, self.memory)
        if bad:
            return fallback
        try:
            a = Action(out) if not isinstance(out, Action) else out
        except Exception:  # noqa: BLE001
            self._err(f"not an action: {out!r}")
            return fallback
        if a not in obs.legal_actions:
            self._err(f"illegal action {a.name}")
            return fallback
        return a

    def hand_end(self, result) -> None:
        if self.forfeited or self.bank_empty:
            return
        self._timed(self.bot.on_hand_end, result, self.memory)


@dataclass
class HalfOut:
    nets_a1: list
    seat_a1: Seat
    seat_a2: Seat


def play_half(spec1: Callable, spec2: Callable, seed: Any, half: int, num_deals: int) -> HalfOut:
    a1 = Seat(spec1())
    a2 = Seat(spec2())
    a1_seat = 0 if half == 1 else 1
    seats = (a1, a2) if a1_seat == 0 else (a2, a1)
    cfg = AgentConfig()
    for s in seats:
        s.start(cfg)
    nets = []
    seed = str(seed)
    for d in range(num_deals):
        g = Game(make_deal(seed, d), button_for_deal(d), hand_index=d)
        while not g.is_terminal:
            st = g.to_act
            g.apply(seats[st].act(g.observation(st)))
        nets.append(g.result().nets[a1_seat])
        for st in (0, 1):
            seats[st].hand_end(g.hand_result(st))
    return HalfOut(nets, a1, a2)


@dataclass
class MatchOut:
    mean_bb: float          # agent1 bb per deal (paired)
    ci: tuple
    verdict: str            # "A1", "A2", "CHOP", "BOTH_FORFEIT"
    points1: float
    points2: float
    halves: tuple

    def faults(self) -> str:
        parts = []
        for h, ho in enumerate(self.halves, 1):
            for name, s in (("a1", ho.seat_a1), ("a2", ho.seat_a2)):
                flags = []
                if s.forfeited:
                    flags.append("FORFEIT")
                if s.bank_empty:
                    flags.append("BANK_EMPTY")
                parts.append(f"h{h}.{name}: {s.time_used:.2f}s/{s.calls}calls err={s.errors}"
                             f" max={s.max_call*1000:.0f}ms {' '.join(flags)}"
                             + (f" [{s.first_error}]" if s.first_error else ""))
        return "\n".join(parts)


def play_match(spec1: Callable, spec2: Callable, seed: Any, num_deals: int = 1000) -> MatchOut:
    h1 = play_half(spec1, spec2, seed, 1, num_deals)
    h2 = play_half(spec1, spec2, seed, 2, num_deals)
    paired = [(h1.nets_a1[d] + h2.nets_a1[d]) / BIG_BLIND for d in range(num_deals)]
    mean = sum(paired) / num_deals
    sd = statistics.stdev(paired) if num_deals > 1 else 0.0
    se = sd / math.sqrt(num_deals)
    ci = (mean - CHOP_Z * se, mean + CHOP_Z * se)
    f1 = h1.seat_a1.forfeited or h2.seat_a1.forfeited
    f2 = h1.seat_a2.forfeited or h2.seat_a2.forfeited
    if f1 and f2:
        v, p1, p2 = "BOTH_FORFEIT", 0.0, 0.0
    elif f1:
        v, p1, p2 = "A2", 0.0, 1.0
    elif f2:
        v, p1, p2 = "A1", 1.0, 0.0
    elif ci[0] > 0:
        v, p1, p2 = "A1", 1.0, 0.0
    elif ci[1] < 0:
        v, p1, p2 = "A2", 0.0, 1.0
    else:
        v, p1, p2 = "CHOP", 0.5, 0.5
    return MatchOut(mean, ci, v, p1, p2, (h1, h2))


# ---------------------------------------------------------------------------
# Wrapping agent.py-style files
# ---------------------------------------------------------------------------

class _ModuleBot:
    def __init__(self, mod):
        self.mod = mod

    def new_half(self, config, memory):
        return self.mod.new_half(config, memory)

    def choose_action(self, obs, memory):
        return self.mod.choose_action(obs, memory)

    def on_hand_end(self, result, memory):
        return self.mod.on_hand_end(result, memory)


def load_agent_file(path: str, fresh_each_half: bool = True) -> Callable:
    """Spec for an agent.py-style file. With fresh_each_half the module is
    re-imported for every half, like the tournament's fresh server process
    (so module-level globals reset too)."""
    path = os.path.abspath(path)
    folder = os.path.dirname(path)

    def _import():
        if folder not in sys.path:
            sys.path.insert(0, folder)
        name = f"_agent_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    cached = {}

    def factory():
        if fresh_each_half or "m" not in cached:
            cached["m"] = _import()
        return _ModuleBot(cached["m"])

    factory.__name__ = os.path.basename(folder) + "/" + os.path.basename(path)
    return factory

"""
hero.py - our bot's strategy (agent.py just calls into this).

Everything that shapes play is a field of Params, so the lab can tune it.
State lives only in `memory` (wiped every half, like the tournament).

How a decision is made:
  1. Range: if the opponent has bet/raised this street, assume it holds the
     top `keep` fraction of hands, where keep comes from how often THIS
     opponent has been aggressive so far (maniacs -> wide, rocks -> narrow).
  2. Equity: Monte Carlo vs that range (fast numpy evaluator).
  3. Facing a bet: raise for value / call if equity beats the pot odds plus a
     margin / sometimes bluff-raise if this opponent over-folds / else fold.
     Not facing a bet: bet for value / bluff sometimes / else check.
  4. Exploit mode: if the opponent folds to almost every bet (e.g. a bot
     whose time bank ran out plays check/fold), raise at every chance.
  5. Time guard: sample count shrinks if we spend faster than our budget.
  6. Any exception -> safe fallback action (never crash).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

import pokerlib as pl
from game import Action, Street

F, C, RH, RP = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
AGGR = (RH, RP)


@dataclass
class Params:
    # sampling
    n_pre: int = 120
    n_post: int = 220
    n_min: int = 48
    time_target: float = 2.6          # real seconds of thinking per half we aim for
    # value thresholds (equity needed) by street: pre, flop, turn, river
    bet_t: tuple = (0.58, 0.60, 0.62, 0.64)
    raise_t: tuple = (0.62, 0.66, 0.68, 0.72)
    pot_t: tuple = (0.72, 0.76, 0.78, 0.82)     # use the pot-size raise above this
    call_margin: float = 0.02         # extra equity over pot odds required to call
    # opponent range model
    range_k: float = 1.6              # keep = clamp(opp_aggr_rate * range_k)
    range_min: float = 0.18
    reraise_shrink: float = 0.7       # each extra raise this street narrows the range
    prior_aggr: float = 0.30          # prior opponent aggression rate
    prior_w: float = 10.0             # weight of the prior (in actions)
    # bluffing
    bluff_base: float = 0.06
    bluff_adapt: float = 0.9          # extra bluff prob per unit of opp fold rate above 0.4
    bluff_floor: float = 0.25         # only bluff with at least this much equity (semi-bluffs)
    thin_value: float = 0.25          # lower bet thresholds vs opponents who rarely fold
    # exploit mode (opponent folds to nearly every bet)
    exploit_window: int = 16
    exploit_folds: int = 14
    # randomization of sizes
    size_mix: float = 0.15            # chance to swap half <-> pot
    # variance control: when ahead (bb/hand) tighten, when behind loosen
    var_ctrl: float = 0.0
    seed: int = 20261107


class Hero:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()

    # ------------------------------------------------------------ state ----
    def new_half(self, config, m):
        m.clear()
        m.update(rng=np.random.default_rng(self.p.seed), t_used=0.0, hands=0, net=0,
                 opp_acts=[0, 0], opp_aggr=[0, 0],      # [preflop, postflop]
                 bet_log=[],                             # 1 = opp folded to our bet, 0 = not
                 hand_seen=0, cur_hand=-1, we_bet_last=False)

    def on_hand_end(self, result, m):
        m["hands"] += 1
        m["net"] += result.my_net
        if m.get("we_bet_last"):
            folded = (not result.went_to_showdown) and result.winner == "me"
            m["bet_log"].append(1 if folded else 0)
            if len(m["bet_log"]) > 60:
                del m["bet_log"][:-60]
        m["we_bet_last"] = False
        m["cur_hand"] = -1

    # ------------------------------------------------------- bookkeeping ----
    def _track(self, obs, m):
        if m["cur_hand"] != obs.hand_index:
            m["cur_hand"] = obs.hand_index
            m["hand_seen"] = 0
        hist = obs.action_history
        for a in hist[m["hand_seen"]:]:
            if not a.is_me:
                g = 0 if a.street == Street.PREFLOP else 1
                m["opp_acts"][g] += 1
                if a.action in AGGR:
                    m["opp_aggr"][g] += 1
                # opponent didn't fold to our bet -> record a 0
                if m["we_bet_last"] and a.action != F:
                    m["bet_log"].append(0)
                    m["we_bet_last"] = False
        m["hand_seen"] = len(hist)

    def _aggr_rate(self, m, g):
        p = self.p
        return (m["opp_aggr"][g] + p.prior_aggr * p.prior_w) / (m["opp_acts"][g] + p.prior_w)

    def _fold_rate(self, m):
        log = m["bet_log"][-30:]
        return (sum(log) + 0.4 * 6) / (len(log) + 6)

    def _exploit_mode(self, m):
        log = m["bet_log"][-self.p.exploit_window:]
        return len(log) >= self.p.exploit_window and sum(log) >= self.p.exploit_folds

    def _samples(self, obs, m):
        p = self.p
        base = p.n_pre if obs.street == Street.PREFLOP else p.n_post
        pace = p.time_target * (obs.hand_index + 1) / 1000.0
        if m["t_used"] > pace * 1.15:
            return p.n_min
        if m["t_used"] > pace:
            return max(p.n_min, base // 2)
        return base

    # ------------------------------------------------------------- act ----
    def choose_action(self, obs, m):
        t0 = time.perf_counter()
        try:
            a = self._decide(obs, m)
            if a not in obs.legal_actions:
                a = C
        except Exception:  # noqa: BLE001 - never crash
            a = C if obs.to_call <= obs.big_blind else F
            if a not in obs.legal_actions:
                a = C
        m["t_used"] = m.get("t_used", 0.0) + (time.perf_counter() - t0)
        if a in AGGR:
            m["we_bet_last"] = True
        return a

    def _pick(self, obs, want, rng):
        legal = obs.legal_actions
        if want in AGGR and self.p.size_mix > 0 and rng.random() < self.p.size_mix:
            want = RP if want == RH else RH
        if want in legal:
            return want
        for alt in (RP, RH):
            if want in AGGR and alt in legal:
                return alt
        return C

    def _decide(self, obs, m):
        p = self.p
        self._track(obs, m)
        rng = m["rng"]
        legal = obs.legal_actions
        can_raise = RH in legal or RP in legal
        st = int(obs.street)
        g = 0 if obs.street == Street.PREFLOP else 1

        # exploit mode: opponent folds to (nearly) everything
        if self._exploit_mode(m) and can_raise:
            return RH if RH in legal else RP

        # variance control: shift all thresholds by running result
        shift = 0.0
        if p.var_ctrl and m["hands"] > 150:
            bb_per_hand = m["net"] / obs.big_blind / m["hands"]
            shift = float(np.clip(bb_per_hand, -1.0, 1.0)) * p.var_ctrl

        facing = obs.to_call > 0
        opp_raised_street = sum(1 for a in obs.action_history
                                if a.street == obs.street and not a.is_me and a.action in AGGR)
        keep = 1.0
        if facing and opp_raised_street:
            keep = min(1.0, max(p.range_min, self._aggr_rate(m, g) * p.range_k))
            keep *= p.reraise_shrink ** max(0, opp_raised_street - 1)
            keep = max(p.range_min * 0.5, keep)
        n = self._samples(obs, m)
        e = pl.equity(obs.hole_cards, obs.board, n, rng, keep)

        fold_rate = self._fold_rate(m)
        thin = p.thin_value * max(0.0, 0.4 - fold_rate)
        bluff_p = p.bluff_base + p.bluff_adapt * max(0.0, fold_rate - 0.4)

        if facing:
            odds = obs.to_call / (obs.pot + obs.to_call)
            if can_raise and e >= p.raise_t[st] + shift:
                return self._pick(obs, RP if e >= p.pot_t[st] + shift else RH, rng)
            if e >= odds + p.call_margin + shift * 0.5:
                return C
            if (can_raise and obs.raises_this_street < 2 and e >= p.bluff_floor
                    and rng.random() < bluff_p * 0.5):
                return self._pick(obs, RP, rng)
            return F if F in legal else C

        # not facing a bet
        if can_raise and e >= p.bet_t[st] - thin + shift:
            return self._pick(obs, RP if e >= p.pot_t[st] + shift else RH, rng)
        if can_raise and e >= p.bluff_floor and rng.random() < bluff_p:
            return self._pick(obs, RH, rng)
        return C


def params_dict(p: Params) -> dict:
    return asdict(p)

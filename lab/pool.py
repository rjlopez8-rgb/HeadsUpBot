"""
lab/pool.py - the opponent pool.

Three families:
  * SIMPLE    - baselines and pure styles (random, station, maniac, check/fold...)
  * AI_TYPICAL- what other teams will most likely get by asking an LLM:
                Monte Carlo equity + pot odds + thresholds, with different
                sample counts / aggression, some with opponent modeling, and
                one that simulates too much and burns its time bank.
  * WEIRD     - odd mechanics meant to break naive bots: check-raisers,
                river-only bluffers, min-raise spammers, preflop shovers,
                an adaptive exploiter that attacks whatever we fold to.

Every bot is a class; POOL maps name -> zero-arg factory (fresh per half).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pokerlib as pl  # noqa: E402
from game import Action, Street  # noqa: E402

F, C, RH, RP = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT


def _pick(obs, *prefs):
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return C


class Base:
    seed = 0

    def new_half(self, config, memory):
        memory["rng"] = np.random.default_rng(self.seed)

    def on_hand_end(self, result, memory):
        pass


# ---------------------------------------------------------------- SIMPLE ----
class Random(Base):
    def choose_action(self, obs, m):
        return obs.legal_actions[int(m["rng"].integers(len(obs.legal_actions)))]


class CheckFold(Base):           # = the starter bot, and = any bot whose time bank ran out
    def choose_action(self, obs, m):
        return C if obs.to_call == 0 else F


class Station(Base):
    def choose_action(self, obs, m):
        return C


class AlwaysPot(Base):
    def choose_action(self, obs, m):
        return _pick(obs, RP, C)


class AlwaysHalf(Base):          # also a "min-raise spammer"
    def choose_action(self, obs, m):
        return _pick(obs, RH, C)


class Maniac(Base):              # raises 70%, otherwise calls, never folds
    def choose_action(self, obs, m):
        r = m["rng"].random()
        if r < 0.35:
            return _pick(obs, RP, C)
        if r < 0.70:
            return _pick(obs, RH, C)
        return C


# ------------------------------------------------------------ AI_TYPICAL ----
class EquityBot(Base):
    """The canonical LLM answer: equity vs random hand, pot odds, thresholds."""

    def __init__(self, n=200, raise_eq=0.65, pot_eq=0.80, margin=0.0, bet_eq=0.60,
                 bluff=0.0, seed=1):
        self.n, self.raise_eq, self.pot_eq, self.margin = n, raise_eq, pot_eq, margin
        self.bet_eq, self.bluff, self.seed = bet_eq, bluff, seed

    def eq(self, obs, m):
        return pl.equity(obs.hole_cards, obs.board, self.n, m["rng"])

    def choose_action(self, obs, m):
        e = self.eq(obs, m)
        if obs.to_call == 0:
            if e > self.pot_eq:
                return _pick(obs, RP, RH, C)
            if e > self.bet_eq:
                return _pick(obs, RH, C)
            if self.bluff and m["rng"].random() < self.bluff:
                return _pick(obs, RH, C)
            return C
        odds = obs.to_call / (obs.pot + obs.to_call)
        if e > self.pot_eq:
            return _pick(obs, RP, RH, C)
        if e > self.raise_eq:
            return _pick(obs, RH, C)
        if e >= odds + self.margin:
            return C
        return F


class SlowEquityBot(EquityBot):
    """Typical 'more simulations = better' bot: 3000 samples per decision.
    In the tournament it runs out of time bank mid-half (then check/folds)."""

    def __init__(self, **kw):
        super().__init__(n=3000, **kw)


class ModelingEquityBot(EquityBot):
    """LLM answer with opponent modeling: tracks opponent aggression and
    tightens/loosens its calling, bluffs more vs frequent folders."""

    def new_half(self, config, m):
        super().new_half(config, m)
        m.update(opp_aggr=0, opp_acts=0, opp_folds=0, our_bets=0)

    def on_hand_end(self, result, m):
        pass

    def choose_action(self, obs, m):
        # count opponent actions from history of this hand (approximate)
        hist = obs.action_history
        if hist and not hist[-1].is_me:
            m["opp_acts"] += 1
            if hist[-1].action in (RH, RP):
                m["opp_aggr"] += 1
        aggr = (m["opp_aggr"] + 1) / (m["opp_acts"] + 3)
        e = self.eq(obs, m)
        adj = (aggr - 0.33) * 0.25          # vs aggressive opp, call lighter
        if obs.to_call == 0:
            if e > self.pot_eq:
                return _pick(obs, RP, RH, C)
            if e > self.bet_eq or m["rng"].random() < 0.12:
                return _pick(obs, RH, C)
            return C
        odds = obs.to_call / (obs.pot + obs.to_call)
        if e > self.pot_eq:
            return _pick(obs, RP, RH, C)
        if e > self.raise_eq - adj:
            return _pick(obs, RH, C)
        if e >= odds - adj:
            return C
        return F


# ----------------------------------------------------------------- WEIRD ----
class CheckRaiser(EquityBot):
    """Checks strong hands to induce bets, then pot-raises. Bets weak hands
    sometimes. Punishes bots that auto-bet when checked to."""

    def choose_action(self, obs, m):
        e = self.eq(obs, m)
        if obs.to_call == 0:
            if e > 0.75:
                return C                                   # trap
            if e < 0.35 and m["rng"].random() < 0.3:
                return _pick(obs, RH, C)                   # weak lead
            return C
        if e > 0.70:
            return _pick(obs, RP, RH, C)
        odds = obs.to_call / (obs.pot + obs.to_call)
        return C if e >= odds else F


class RiverBluffer(EquityBot):
    """Passive until the river, then pot-raises 60% of the time no matter what."""

    def choose_action(self, obs, m):
        if obs.street == Street.RIVER and m["rng"].random() < 0.6:
            return _pick(obs, RP, C)
        e = self.eq(obs, m)
        if obs.to_call == 0:
            return C
        odds = obs.to_call / (obs.pot + obs.to_call)
        return C if e >= odds else F


class PreflopShover(EquityBot):
    """Pot-raises every preflop decision (gets all-in fast), then plays equity."""

    def choose_action(self, obs, m):
        if obs.street == Street.PREFLOP:
            return _pick(obs, RP, C)
        return super().choose_action(obs, m)


class Exploiter(Base):
    """Adaptive red-team bot. Measures how often WE fold when it bets/raises,
    and how often we bet. Bluffs relentlessly if we over-fold, value-bets
    thin if we over-call, and calls down light if we bluff a lot."""

    seed = 7

    def new_half(self, config, m):
        super().new_half(config, m)
        m.update(bets=0, folds_to_bet=0, opp_bets=0, opp_acts=0, last=None)

    def on_hand_end(self, result, m):
        # if our last action this hand was a bet and the hand ended without showdown
        # and we won -> opponent folded to our bet
        if m["last"] == "bet":
            m["bets"] += 1
            if not result.went_to_showdown and result.winner == "me":
                m["folds_to_bet"] += 1
        m["last"] = None

    def choose_action(self, obs, m):
        rng = m["rng"]
        hist = obs.action_history
        if hist and not hist[-1].is_me:
            m["opp_acts"] += 1
            if hist[-1].action in (RH, RP):
                m["opp_bets"] += 1
        fold_rate = (m["folds_to_bet"] + 1) / (m["bets"] + 2)
        opp_aggr = (m["opp_bets"] + 1) / (m["opp_acts"] + 3)
        e = pl.equity(obs.hole_cards, obs.board, 150, rng)
        bluff_p = max(0.0, (fold_rate - 0.35) * 2.0)          # 0 at 35% fold, 1 at 85%
        if obs.to_call == 0:
            if e > 0.62 or rng.random() < bluff_p:
                m["last"] = "bet"
                return _pick(obs, RP if e > 0.8 or fold_rate > 0.6 else RH, C)
            return C
        odds = obs.to_call / (obs.pot + obs.to_call)
        if e > 0.75 or (rng.random() < bluff_p * 0.6 and obs.raises_this_street < 2):
            m["last"] = "bet"
            return _pick(obs, RP, RH, C)
        light = (opp_aggr - 0.3) * 0.3                         # call lighter vs aggro
        return C if e >= odds - light else F


# ------------------------------------------------- "LLM straight from the guide"
class GuideExample(Base):
    """The example strategy printed in the participant guide (section 4.1)."""

    def choose_action(self, obs, m):
        from evaluator import FLUSH, omaha_score
        if obs.street != Street.RIVER:
            if obs.to_call <= obs.big_blind:
                return C
            if F in obs.legal_actions:
                return F
            return C
        score = omaha_score(obs.hole_cards, obs.board)
        if score >= (FLUSH << 20) and RP in obs.legal_actions:
            return RP
        return C


class OfficialMC(Base):
    """Typical LLM bot following the guide's tips literally: Monte Carlo with
    the OFFICIAL evaluator.omaha_score (slow), pot odds, preflop features.
    With sims=150 it burns ~8 ms per decision -> its time bank runs out."""

    def __init__(self, sims=150, seed=11, raise_eq=0.68, bet_eq=0.62):
        self.sims, self.seed, self.raise_eq, self.bet_eq = sims, seed, raise_eq, bet_eq

    def new_half(self, config, m):
        import random
        m["r"] = random.Random(self.seed)

    def _eq(self, obs, m):
        from evaluator import omaha_score
        r = m["r"]
        dead = set(obs.hole_cards) | set(obs.board)
        deck = [c for c in range(36) if c not in dead]
        need = 5 - len(obs.board)
        win = 0.0
        for _ in range(self.sims):
            d = r.sample(deck, 5 + need)
            board = list(obs.board) + d[5:]
            a, b = omaha_score(obs.hole_cards, board), omaha_score(d[:5], board)
            win += 1.0 if a > b else 0.5 if a == b else 0.0
        return win / self.sims

    def _preflop(self, obs):
        ranks = [c >> 2 for c in obs.hole_cards]
        suits = [c & 3 for c in obs.hole_cards]
        score = sum(ranks) / 40 + 0.15 * (len(ranks) - len(set(ranks))) + \
            0.1 * sum(1 for s in set(suits) if suits.count(s) >= 2)
        return score

    def choose_action(self, obs, m):
        if obs.street == Street.PREFLOP:
            s = self._preflop(obs)
            if s > 0.75:
                return _pick(obs, RP, C)
            if s > 0.45 or obs.to_call <= obs.big_blind:
                return C
            return F if obs.to_call > 0 else C
        e = self._eq(obs, m)
        if obs.to_call == 0:
            return _pick(obs, RH, C) if e > self.bet_eq else C
        odds = obs.to_call / (obs.pot + obs.to_call)
        if e > self.raise_eq:
            return _pick(obs, RP, RH, C)
        return C if e > odds else F


SIMPLE = {
    "random": Random, "checkfold": CheckFold, "station": Station,
    "always_pot": AlwaysPot, "always_half": AlwaysHalf, "maniac": Maniac,
}
AI_TYPICAL = {
    "eq_basic": lambda: EquityBot(),
    "eq_tight": lambda: EquityBot(raise_eq=0.72, pot_eq=0.85, margin=0.05, bet_eq=0.66, seed=2),
    "eq_loose_aggr": lambda: EquityBot(raise_eq=0.55, pot_eq=0.70, margin=-0.05, bet_eq=0.50, bluff=0.15, seed=3),
    "eq_modeling": lambda: ModelingEquityBot(seed=4),
    "eq_slow": lambda: SlowEquityBot(seed=5),
    "guide_example": GuideExample,
    "official_mc": lambda: OfficialMC(),
    "official_mc_fast": lambda: OfficialMC(sims=40, seed=12),
}
WEIRD = {
    "check_raiser": lambda: CheckRaiser(seed=6),
    "river_bluffer": lambda: RiverBluffer(seed=8),
    "pf_shover": lambda: PreflopShover(seed=9),
    "exploiter": Exploiter,
}
POOL = {**SIMPLE, **AI_TYPICAL, **WEIRD}

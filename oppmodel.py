"""
oppmodel.py - spot-specific opponent statistics.

The overall raise rate hides the leaks that matter: an opponent can raise
about as often as a solver overall and still 3-bet far less (so its 3-bets
are much stronger) or never fold to a 4-bet. So every opponent decision is
counted per SPOT:

    spot = (street, raises already made this street (0..3+), facing a bet?)

e.g. (0, 1, True)  = preflop, facing one raise    -> their 3-bet decision
     (0, 2, True)  = preflop, facing a 3-bet       -> their 4-bet decision
     (3, 0, False) = river, checked to / first     -> their river bet decision

Decisions that END a hand never show up in a later observation (a fold to our
bet, a call of our river bet), so on_hand_end infers them from the result.
"""

from __future__ import annotations

from game import Action, Street

F, C, RH, RP = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
AGGR = (RH, RP)


def spots(history):
    """Yield (index, spot, action, is_me) for every action in a hand history."""
    raises = {0: 0, 1: 0, 2: 0, 3: 0}
    first_pre = True
    pending = {0: False, 1: False, 2: False, 3: False}   # is there a bet to answer?
    for i, a in enumerate(history):
        s = int(a.street)
        if s == 0 and first_pre:
            facing = True               # the button faces the big blind
            first_pre = False
        else:
            facing = pending[s]
        yield i, (s, min(raises[s], 3), facing), a.action, a.is_me
        if a.action in AGGR:
            raises[s] += 1
            pending[s] = True
        elif a.action == C:
            pending[s] = False


def _bump(stats, spot, action):
    rec = stats.setdefault(spot, [0, 0, 0])
    rec[0 if action == F else 2 if action in AGGR else 1] += 1


class OppModel:
    """Lives inside the bot's memory dict (so it resets every half)."""

    @staticmethod
    def init(m):
        m["om"] = {}            # spot -> [folds, calls/checks, raises]
        m["om_hand"] = -1
        m["om_done"] = 0
        m["om_hist"] = ()
        m["om_last"] = None     # (spot after our last action, our action)

    @staticmethod
    def observe(m, obs):
        if m["om_hand"] != obs.hand_index:
            m["om_hand"], m["om_done"] = obs.hand_index, 0
        hist = obs.action_history
        for i, spot, act, is_me in spots(hist):
            if i >= m["om_done"] and not is_me:
                _bump(m["om"], spot, act)
        m["om_done"] = len(hist)
        m["om_hist"] = hist

    @staticmethod
    def our_action(m, obs, action):
        """Remember the spot the OPPONENT faces after our action."""
        s = int(obs.street)
        raises = obs.raises_this_street + (1 if action in AGGR else 0)
        facing = action in AGGR
        m["om_last"] = ((s, min(raises, 3), facing), action)

    @staticmethod
    def hand_end(m, result):
        if m.get("om_hand") != result.hand_index or m.get("om_last") is None:
            m["om_last"] = None
            return
        spot, ours = m["om_last"]
        if ours in AGGR:
            if not result.went_to_showdown and result.winner == "me":
                _bump(m["om"], spot, F)          # they folded to our bet/raise
            elif result.went_to_showdown:
                _bump(m["om"], spot, C)          # they called our (last) bet or all-in
        m["om_last"] = None

    @staticmethod
    def freq(m, spot):
        """(n, fold %, call %, raise %) for a spot, or (0, None...)."""
        rec = m["om"].get(spot)
        if not rec or sum(rec) == 0:
            return 0, None, None, None
        n = sum(rec)
        return n, rec[0] / n, rec[1] / n, rec[2] / n

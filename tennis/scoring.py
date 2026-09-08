"""
Exact probabilities for the tennis scoring tree.

Tennis is the rare sport whose scoring is a closed, finite Markov chain.
Given only two numbers -- P(A wins a point on A's serve) and P(B wins a
point on B's serve) -- every downstream probability is EXACT arithmetic:
this game, this tiebreak, this set, the match, the total number of games,
the correct set score. No Monte Carlo, no sampling error.

That is the whole reason this module is separate from inference.py.
The hard, uncertain part of the problem is estimating those two point
probabilities from what we observe (inference.py's job). Everything
downstream of them is arithmetic we can do perfectly.

Convention throughout:
    player 0 = "A",  player 1 = "B"
    pa = P(A wins a point when A is serving)
    pb = P(B wins a point when B is serving)
"""
from dataclasses import dataclass, field
from collections import defaultdict


# =====================================================================
# Match format
# =====================================================================

@dataclass
class MatchFormat:
    """The rules that shape the scoring tree."""
    best_of: int = 3              # 3 or 5 sets
    set_games: int = 6            # games needed to win a set
    tiebreak_at: int = 6          # games-all at which a tiebreak starts
    tiebreak_points: int = 7      # points to win a normal-set tiebreak
    final_set: str = "tb7"        # "tb7" | "tb10" | "advantage" | "super_set"
    adv_cap: int = 25             # games cap for advantage final sets

    @property
    def sets_to_win(self):
        return self.best_of // 2 + 1

    def is_final_set(self, set_index):
        return set_index == self.best_of - 1

    def tb_points_for_set(self, set_index):
        """Points needed to win the tiebreak of this set, or None if the set
        is played to advantage with no tiebreak at all."""
        if self.is_final_set(set_index):
            if self.final_set in ("tb10", "super_set"):
                return 10
            if self.final_set == "advantage":
                return None
        return self.tiebreak_points

    def is_super_set(self, set_index):
        """True if this whole 'set' is a single 10-point tiebreak
        (doubles / mixed deciders)."""
        return self.final_set == "super_set" and self.is_final_set(set_index)


BO3 = MatchFormat(best_of=3, final_set="tb7")
BO3_SUPER = MatchFormat(best_of=3, final_set="super_set")
BO5 = MatchFormat(best_of=5, final_set="tb7")
BO5_TB10 = MatchFormat(best_of=5, final_set="tb10")
BO5_ADV = MatchFormat(best_of=5, final_set="advantage")

FORMATS = {
    "bo3": BO3, "bo3super": BO3_SUPER, "bo5": BO5,
    "bo5tb10": BO5_TB10, "bo5adv": BO5_ADV,
}


# =====================================================================
# Live score state
# =====================================================================

POINT_LABELS = {0: "0", 1: "15", 2: "30", 3: "40"}


@dataclass
class Score:
    """Complete live state of a match, advanced one point at a time."""
    sets: list = field(default_factory=lambda: [0, 0])
    games: list = field(default_factory=lambda: [0, 0])
    points: list = field(default_factory=lambda: [0, 0])
    server: int = 0                 # who serves the next point
    in_tiebreak: bool = False
    tb_left: int = 1                # points left in this server's tiebreak block
    tb_first_server: int = 0        # who served the first point of the tiebreak
    set_index: int = 0              # 0-based
    winner: int = None              # set once the match is over
    set_scores: list = field(default_factory=list)   # [(ga, gb), ...] finished
    total_games: int = 0            # games completed across the whole match
    points_played: int = 0

    def copy(self):
        s = Score(**{k: (list(v) if isinstance(v, list) else v)
                     for k, v in self.__dict__.items()})
        return s

    # -- advancing ------------------------------------------------------
    def award_point(self, winner, fmt=BO3):
        """Give one point to `winner` (0 or 1). Returns a list of event strings."""
        if self.winner is not None:
            return []
        events = []
        self.points[winner] += 1
        self.points_played += 1

        if self.in_tiebreak:
            target = fmt.tb_points_for_set(self.set_index) or fmt.tiebreak_points
            # serve rotation inside a tiebreak: 1 point, then blocks of 2
            if self.tb_left > 1:
                self.tb_left -= 1
            else:
                self.server = 1 - self.server
                self.tb_left = 2
            if _decided(self.points, target, 2):
                w = 0 if self.points[0] > self.points[1] else 1
                events.append(f"tiebreak {self.points[0]}-{self.points[1]} to {'AB'[w]}")
                self.games[w] += 1
                self.total_games += 1
                self._close_set(w, fmt, events, after_tiebreak=True)
            return events

        # ordinary game
        if _decided(self.points, 4, 2):
            w = 0 if self.points[0] > self.points[1] else 1
            self.games[w] += 1
            self.total_games += 1
            self.points = [0, 0]
            held = (w == self.server)
            events.append(f"{'game' if held else 'BREAK'} {'AB'[w]}"
                          f"  ({self.games[0]}-{self.games[1]})")
            self.server = 1 - self.server
            self._check_set_end(fmt, events)
        return events

    def _check_set_end(self, fmt, events):
        ga, gb = self.games
        tb_pts = fmt.tb_points_for_set(self.set_index)
        if max(ga, gb) >= fmt.set_games and abs(ga - gb) >= 2:
            self._close_set(0 if ga > gb else 1, fmt, events)
        elif tb_pts is not None and ga == fmt.tiebreak_at and gb == fmt.tiebreak_at:
            self._start_tiebreak(tb_pts, events)
        elif tb_pts is None and max(ga, gb) >= fmt.adv_cap:
            self._close_set(0 if ga > gb else 1, fmt, events)

    def _start_tiebreak(self, target, events):
        self.in_tiebreak = True
        self.tb_left = 1
        self.tb_first_server = self.server
        self.points = [0, 0]
        events.append(f"-- tiebreak, first to {target} --")

    def _close_set(self, w, fmt, events, after_tiebreak=False):
        self.sets[w] += 1
        self.set_scores.append((self.games[0], self.games[1]))
        events.append(f"SET {'AB'[w]}  {self.games[0]}-{self.games[1]}"
                      f"  (sets {self.sets[0]}-{self.sets[1]})")
        if after_tiebreak:
            # whoever received the first tiebreak point serves the new set
            self.server = 1 - self.tb_first_server
        self.in_tiebreak = False
        self.tb_left = 1
        self.games = [0, 0]
        self.points = [0, 0]
        self.set_index += 1
        if self.sets[w] >= fmt.sets_to_win:
            self.winner = w
            events.append(f"MATCH {'AB'[w]}")
        elif fmt.is_super_set(self.set_index):
            self._start_tiebreak(10, events)

    # -- display --------------------------------------------------------
    def point_str(self):
        a, b = self.points
        if self.in_tiebreak:
            return f"{a}-{b}"
        if a >= 3 and b >= 3:
            if a == b:
                return "40-40"
            return "AD-40" if a > b else "40-AD"
        return f"{POINT_LABELS.get(a, '40')}-{POINT_LABELS.get(b, '40')}"

    def line(self, name_a="A", name_b="B"):
        done = "  ".join(f"{x}-{y}" for x, y in self.set_scores)
        srv_a = "*" if self.server == 0 and not self.winner else ""
        srv_b = "*" if self.server == 1 and not self.winner else ""
        cur = f"{self.games[0]}-{self.games[1]}"
        return (f"{name_a}{srv_a} {self.sets[0]}-{self.sets[1]} {name_b}{srv_b}"
                f"  |  {done + '   ' if done else ''}{cur}"
                f"  |  {self.point_str()}")


def _decided(pts, target, margin):
    return max(pts) >= target and abs(pts[0] - pts[1]) >= margin


# =====================================================================
# Level 1: the game
# =====================================================================

def game_win_prob(p, server_pts=0, returner_pts=0):
    """P(server wins this game) given point-win prob p and the current score.

    Deuce is a cycle in the state graph, so it takes the closed form
    p^2 / (p^2 + q^2) rather than a recursion that would never bottom out.
    """
    if server_pts >= 4 and server_pts - returner_pts >= 2:
        return 1.0
    if returner_pts >= 4 and returner_pts - server_pts >= 2:
        return 0.0
    q = 1.0 - p
    denom = p * p + q * q
    p_deuce = (p * p / denom) if denom > 0 else 0.5

    if server_pts >= 3 and returner_pts >= 3:
        d = server_pts - returner_pts
        if d == 0:
            return p_deuce                  # deuce
        if d > 0:
            return p + q * p_deuce          # advantage server
        return p * p_deuce                  # advantage returner

    return (p * game_win_prob(p, server_pts + 1, returner_pts)
            + q * game_win_prob(p, server_pts, returner_pts + 1))


def hold_prob(p):
    """Probability of holding serve from 0-0."""
    return game_win_prob(p, 0, 0)


# =====================================================================
# Level 2: the tiebreak
# =====================================================================

_TB_DEUCE_CACHE = {}


def _tb_deuce_values(pa, pb):
    """Solve the tiebreak deuce cycle by value iteration.

    Past 6-6 the future depends only on (point difference, who serves, how
    many points remain in their block) -- 12 states. Serve rotation makes
    this a genuine cycle, so we iterate to a fixed point instead of
    recursing. Contraction is at worst 1/2 per point, so 400 sweeps puts
    us far below float precision.
    """
    key = (round(pa, 12), round(pb, 12))
    if key in _TB_DEUCE_CACHE:
        return _TB_DEUCE_CACHE[key]
    keys = [(d, s, l) for d in (-1, 0, 1) for s in (0, 1) for l in (1, 2)]
    V = {k: 0.5 for k in keys}
    for _ in range(400):
        nV = {}
        for (d, s, l) in keys:
            q = pa if s == 0 else 1.0 - pb          # P(A wins this point)
            ns, nl = (s, l - 1) if l > 1 else (1 - s, 2)
            up = 1.0 if d + 1 >= 2 else V[(d + 1, ns, nl)]
            dn = 0.0 if d - 1 <= -2 else V[(d - 1, ns, nl)]
            nV[(d, s, l)] = q * up + (1.0 - q) * dn
        V = nV
    _TB_DEUCE_CACHE[key] = V
    return V


def tiebreak_win_prob(pa, pb, a=0, b=0, server=0, left=1, target=7):
    """P(A wins the tiebreak) from any point score and serve position."""
    V = _tb_deuce_values(pa, pb)
    memo = {}

    def rec(a, b, s, l):
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        if a >= target - 1 and b >= target - 1:
            return V[(a - b, s, l)]
        key = (a, b, s, l)
        if key in memo:
            return memo[key]
        q = pa if s == 0 else 1.0 - pb
        ns, nl = (s, l - 1) if l > 1 else (1 - s, 2)
        r = q * rec(a + 1, b, ns, nl) + (1.0 - q) * rec(a, b + 1, ns, nl)
        memo[key] = r
        return r

    return rec(a, b, server, left)


# =====================================================================
# Levels 3 & 4: the set and the match
# =====================================================================

class MatchModel:
    """Composes point probabilities up through game, set and match.

    Every method returns exact probabilities. `outcome_dist` returns the
    full joint distribution over (winner, sets A, sets B, total games),
    which is what the betting markets are priced off.
    """

    def __init__(self, pa, pb, fmt=BO3):
        self.pa = min(max(pa, 1e-6), 1 - 1e-6)
        self.pb = min(max(pb, 1e-6), 1 - 1e-6)
        self.fmt = fmt
        self.hold_a = hold_prob(self.pa)
        self.hold_b = hold_prob(self.pb)
        self._set_memo = {}
        self._match_memo = {}

    # -- current game ---------------------------------------------------
    def game_prob(self, score):
        """P(A wins the game (or tiebreak) currently in progress)."""
        if score.in_tiebreak:
            target = (self.fmt.tb_points_for_set(score.set_index)
                      or self.fmt.tiebreak_points)
            return tiebreak_win_prob(self.pa, self.pb, score.points[0],
                                     score.points[1], score.server,
                                     score.tb_left, target)
        if score.server == 0:
            return game_win_prob(self.pa, score.points[0], score.points[1])
        return 1.0 - game_win_prob(self.pb, score.points[1], score.points[0])

    # -- a full set from a games score ----------------------------------
    def _set_dist(self, set_index, ga, gb, server):
        """{(winner, next_server, games_from_here): prob} for a set resumed
        at a games score with no game in progress."""
        key = (set_index, ga, gb, server)
        if key in self._set_memo:
            return self._set_memo[key]
        fmt = self.fmt
        tb_pts = fmt.tb_points_for_set(set_index)
        out = defaultdict(float)

        if max(ga, gb) >= fmt.set_games and abs(ga - gb) >= 2:
            out[(0 if ga > gb else 1, server, 0)] = 1.0
        elif tb_pts is not None and ga == fmt.tiebreak_at and gb == fmt.tiebreak_at:
            tb = tiebreak_win_prob(self.pa, self.pb, 0, 0, server, 1, tb_pts)
            nxt = 1 - server            # first receiver of the TB serves next set
            out[(0, nxt, 1)] = tb
            out[(1, nxt, 1)] = 1.0 - tb
        elif tb_pts is None and max(ga, gb) >= fmt.adv_cap:
            out[(0 if ga > gb else 1, server, 0)] = 1.0
        else:
            wa = self.hold_a if server == 0 else 1.0 - self.hold_b
            for winner_of_game, p in ((0, wa), (1, 1.0 - wa)):
                if p <= 0:
                    continue
                nga = ga + (1 if winner_of_game == 0 else 0)
                ngb = gb + (1 if winner_of_game == 1 else 0)
                for (w, ns, g), q in self._set_dist(set_index, nga, ngb,
                                                    1 - server).items():
                    out[(w, ns, g + 1)] += p * q

        out = dict(out)
        self._set_memo[key] = out
        return out

    def _current_set_dist(self, score):
        """Same, but resumed mid-game or mid-tiebreak from the live score."""
        fmt = self.fmt
        pg = self.game_prob(score)
        out = defaultdict(float)

        if score.in_tiebreak:
            nxt = 1 - score.tb_first_server
            out[(0, nxt, 1)] += pg
            out[(1, nxt, 1)] += 1.0 - pg
            return dict(out)

        for winner_of_game, p in ((0, pg), (1, 1.0 - pg)):
            if p <= 0:
                continue
            nga = score.games[0] + (1 if winner_of_game == 0 else 0)
            ngb = score.games[1] + (1 if winner_of_game == 1 else 0)
            for (w, ns, g), q in self._set_dist(score.set_index, nga, ngb,
                                                1 - score.server).items():
                out[(w, ns, g + 1)] += p * q
        return dict(out)

    # -- the match ------------------------------------------------------
    def _match_dist(self, sa, sb, set_index, server):
        """{(winner, sets_a, sets_b, games_from_here): prob} for whole sets."""
        key = (sa, sb, set_index, server)
        if key in self._match_memo:
            return self._match_memo[key]
        need = self.fmt.sets_to_win
        out = defaultdict(float)
        if sa >= need or sb >= need:
            out[(0 if sa > sb else 1, sa, sb, 0)] = 1.0
        else:
            for (w, ns, g), p in self._set_dist(set_index, 0, 0, server).items():
                nsa = sa + (1 if w == 0 else 0)
                nsb = sb + (1 if w == 1 else 0)
                for (mw, fa, fb, gg), q in self._match_dist(
                        nsa, nsb, set_index + 1, ns).items():
                    out[(mw, fa, fb, g + gg)] += p * q
        out = dict(out)
        self._match_memo[key] = out
        return out

    def outcome_dist(self, score):
        """Full joint distribution from the live score onward.

        Keys are (winner, final sets A, final sets B, total games in the
        whole match, counting games already played).
        """
        if score.winner is not None:
            return {(score.winner, score.sets[0], score.sets[1],
                     score.total_games): 1.0}
        played = score.total_games
        out = defaultdict(float)
        for (w, ns, g), p in self._current_set_dist(score).items():
            nsa = score.sets[0] + (1 if w == 0 else 0)
            nsb = score.sets[1] + (1 if w == 1 else 0)
            for (mw, fa, fb, gg), q in self._match_dist(
                    nsa, nsb, score.set_index + 1, ns).items():
                out[(mw, fa, fb, played + g + gg)] += p * q
        return dict(out)

    # -- convenience readouts -------------------------------------------
    def set_prob(self, score):
        """P(A wins the set currently in progress)."""
        if score.winner is not None:
            return float(score.winner == 0)
        return sum(p for (w, _, _), p in self._current_set_dist(score).items()
                   if w == 0)

    def match_prob(self, score):
        """P(A wins the match)."""
        return sum(p for (w, _, _, _), p in self.outcome_dist(score).items()
                   if w == 0)

    def set_score_dist(self, score, top=6):
        """Most likely final set scores, e.g. ((2, 0), 0.41)."""
        agg = defaultdict(float)
        for (w, fa, fb, _), p in self.outcome_dist(score).items():
            agg[(fa, fb)] += p
        return sorted(agg.items(), key=lambda kv: -kv[1])[:top]

    def total_games_dist(self, score):
        agg = defaultdict(float)
        for (_, _, _, g), p in self.outcome_dist(score).items():
            agg[g] += p
        return dict(sorted(agg.items()))

    def over_prob(self, score, line):
        """P(total games in the match > line). Use a .5 line to avoid pushes."""
        return sum(p for g, p in self.total_games_dist(score).items() if g > line)

    def expected_total_games(self, score):
        return sum(g * p for g, p in self.total_games_dist(score).items())

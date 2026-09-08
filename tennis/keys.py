"""
The input layer: two keys for hits, eight for outcomes.

    a . . . . . . . . l          <-- HIT keys (home row)

    q w                o p       <-- ABOVE = that player WON the point
    a                    l
    z x              , . /       <-- BELOW = that player LOST the point

While the rally is live you tap `a` every time the left player strikes the
ball and `l` every time the right player strikes it. When the point ends
you tap any key above or below your player's home key.

Because "A won" and "L lost" are the same event, the eight outcome keys
collapse to two verdicts. The redundancy is the point: whichever hand is
free ends the point.

Nothing else needs to be typed. The key stream already contains everything
we need to classify the point, because we know WHO TOUCHED THE BALL LAST:

    last hitter == point winner  ->  that shot was a WINNER
    last hitter != point winner  ->  that shot was an ERROR

and the rally length falls out of the count of hits:

    aw      1 hit,  A won   ->  A's serve untouched      (ace / service winner)
    az      1 hit,  A lost  ->  A's serve never landed   (double fault)
    al.     2 hits, A won   ->  L's return missed        (return error)
    alo     2 hits, L won   ->  L's return was a winner
    alaw    3 hits, A won   ->  A's serve+1 was a winner
    alalz   4 hits, L won   ->  L's 4th ball forced/drew the miss ... etc.

Every keystroke is also timestamped, so inter-hit intervals give ball pace
and court position for free, and the gap between points measures fatigue.
"""
import time
from dataclasses import dataclass, field


# =====================================================================
# Key map
# =====================================================================

HIT_KEYS = {"a": 0, "l": 1}

# outcome key -> index of the player who WON the point
OUTCOME_KEYS = {
    "q": 0, "w": 0,           # above `a`  -> A won
    "z": 1, "x": 1,           # below `a`  -> A lost, so B won
    "o": 1, "p": 1,           # above `l`  -> B won
    ",": 0, ".": 0, "/": 0,   # below `l`  -> B lost, so A won
}

FAULT_KEY = "f"       # optional: first serve missed, point continues on 2nd
LET_KEY = "t"         # optional: let / replay, discard the point

PLAYER_KEY = {0: "a", 1: "l"}


# =====================================================================
# One observed point
# =====================================================================

@dataclass
class PointRecord:
    """Everything we observed about a single point."""
    server: int
    hitters: list = field(default_factory=list)   # player index per shot, in order
    times: list = field(default_factory=list)     # perf_counter stamp per shot
    winner: int = 0
    serve_number: int = 1                          # 1 or 2 (needs the `f` key)
    keys: str = ""                                 # raw keystrokes, for replay
    # score context at the moment the point STARTED, for pressure analysis
    set_index: int = 0
    games: tuple = (0, 0)
    points: tuple = (0, 0)
    in_tiebreak: bool = False

    @property
    def rally_length(self):
        return len(self.hitters)

    @property
    def last_hitter(self):
        return self.hitters[-1] if self.hitters else None

    @property
    def ended_by(self):
        """'winner' if the last player to touch the ball won the point,
        'error' if they lost it. Derived, never typed."""
        if not self.hitters:
            return "unknown"
        return "winner" if self.winner == self.last_hitter else "error"

    @property
    def intervals(self):
        """Seconds between consecutive strikes. Short = flat and fast."""
        return [b - a for a, b in zip(self.times, self.times[1:])]

    def describe(self):
        n = self.rally_length
        end = self.ended_by
        who = "AL"[self.last_hitter] if self.hitters else "?"
        if n == 1:
            return "ace / service winner" if end == "winner" else "double fault"
        if n == 2:
            return f"return {'winner' if end == 'winner' else 'error'}"
        kind = "winner" if end == "winner" else "error"
        return f"{n}-shot rally, {who} {kind} on shot {n}"


# =====================================================================
# Incremental builder -- fed one keystroke at a time
# =====================================================================

class PointBuilder:
    """Accumulates keystrokes until a point resolves.

    feed(key) returns a PointRecord when the point ends, None while it is
    still live, and raises nothing on junk keys -- they are ignored.
    """

    def __init__(self, server, clock=time.perf_counter, **context):
        self.server = server
        self.clock = clock
        self.context = context
        self.hitters = []
        self.times = []
        self.keys = ""
        self.serve_number = 1
        self.faults = 0

    def feed(self, key):
        key = key.lower()

        if key == FAULT_KEY:
            # first serve missed: the serve we already logged didn't count
            self.faults += 1
            self.serve_number = min(2, 1 + self.faults)
            if self.hitters:
                self.hitters.pop()
                self.times.pop()
            self.keys += key
            return None

        if key in HIT_KEYS:
            self.hitters.append(HIT_KEYS[key])
            self.times.append(self.clock())
            self.keys += key
            return None

        if key in OUTCOME_KEYS:
            self.keys += key
            return self._finish(OUTCOME_KEYS[key])

        return None

    def undo_key(self):
        """Remove the last keystroke. Returns True if anything was removed."""
        if not self.keys:
            return False
        last = self.keys[-1]
        self.keys = self.keys[:-1]
        if last in HIT_KEYS and self.hitters:
            self.hitters.pop()
            self.times.pop()
        elif last == FAULT_KEY:
            self.faults = max(0, self.faults - 1)
            self.serve_number = min(2, 1 + self.faults)
        return True

    def _finish(self, winner):
        if not self.hitters:
            # outcome pressed with no hits logged: treat as a serve that
            # never came back (ace or double fault) by the known server
            self.hitters = [self.server]
            self.times = [self.clock()]
        return PointRecord(
            server=self.server,
            hitters=list(self.hitters),
            times=list(self.times),
            winner=winner,
            serve_number=self.serve_number,
            keys=self.keys,
            **self.context,
        )

    @property
    def live_str(self):
        if not self.hitters:
            return "(serve)"
        return "".join(PLAYER_KEY[h] for h in self.hitters)


# =====================================================================
# Batch parsing -- for replaying a logged match or typing points offline
# =====================================================================

def parse_point(keystring, server, clock=None, **context):
    """Parse a whole point from a key string, e.g. 'alalw'."""
    tick = [0.0]

    def fake_clock():
        tick[0] += 1.0
        return tick[0]

    b = PointBuilder(server, clock=clock or fake_clock, **context)
    for ch in keystring.strip():
        rec = b.feed(ch)
        if rec is not None:
            return rec
    return None


def parse_stream(text, first_server=0, fmt=None):
    """Parse whitespace-separated point strings into records, advancing the
    score so each point knows who was serving. Returns (records, score)."""
    from tennis.scoring import Score, BO3
    fmt = fmt or BO3
    score = Score(server=first_server)
    records = []
    for tok in text.split():
        rec = parse_point(
            tok, score.server,
            set_index=score.set_index,
            games=tuple(score.games),
            points=tuple(score.points),
            in_tiebreak=score.in_tiebreak,
        )
        if rec is None:
            continue
        records.append(rec)
        score.award_point(rec.winner, fmt)
        if score.winner is not None:
            break
    return records, score

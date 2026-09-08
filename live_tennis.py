#!/usr/bin/env python3
"""
Live Tennis Scorer
====================
Type each point as it happens using the two-hand key scheme from
tennis/keys.py, and watch the score, point-by-point, plus a live win
probability computed exactly from tennis/scoring.py's Markov-chain model.

    a . . . . . . . . l          <-- HIT keys (home row)

    q w                o p       <-- ABOVE = that player WON the point
    a                    l
    z x              , . /       <-- BELOW = that player LOST the point

Tap `a` every time player A strikes the ball, `l` every time player B
strikes it, then finish the point with one outcome key. Example: "aw" is
an ace (A served, A won with 1 hit). "alz" is a service return winner
for B (A served, L returned, A "lost" -> B won).

Usage:
    python live_tennis.py              (interactive)
    python live_tennis.py --demo       (auto-run a demo match)
"""
import sys

from tennis.keys import PointBuilder, OUTCOME_KEYS
from tennis.scoring import Score, MatchModel, FORMATS, BO3


KEY_HELP = """
  Commands:
    <keys>      type a point's keystrokes, e.g. aw / al. / alalz
    undo        remove the last keystroke you typed for the point in progress
    predict     show live win probability / likely set scores
    score       reprint the current score line
    quit        stop and print a summary
"""


def run_interactive():
    print()
    print("  =============================================")
    print("  LIVE TENNIS SCORER - First Principles")
    print("  =============================================")

    name_a = input("\n  Player A name (serves first): ").strip() or "A"
    name_b = input("  Player B name: ").strip() or "B"

    fmt_key = input("  Format [bo3/bo3super/bo5/bo5tb10/bo5adv] (default bo3): ").strip().lower()
    fmt = FORMATS.get(fmt_key, BO3)

    print(f"\n  {name_a} vs {name_b}  ({fmt_key or 'bo3'})")
    print(KEY_HELP)

    score = Score(server=0)
    records = []
    server_pts = [[0, 0], [0, 0]]   # server_pts[server][0]=won, [1]=lost

    print(f"  {score.line(name_a, name_b)}")

    while score.winner is None:
        builder = PointBuilder(score.server, set_index=score.set_index,
                                games=tuple(score.games), points=tuple(score.points),
                                in_tiebreak=score.in_tiebreak)
        while True:
            prompt = f"  [{score.server == 0 and name_a or name_b} to serve] {builder.live_str} > "
            try:
                line = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                _summary(records, score, name_a, name_b)
                return

            if not line:
                continue
            cmd = line.lower()

            if cmd == "quit":
                _summary(records, score, name_a, name_b)
                return
            if cmd == "undo":
                builder.undo_key()
                continue
            if cmd == "score":
                print(f"  {score.line(name_a, name_b)}")
                continue
            if cmd == "predict":
                _predict(server_pts, score, name_a, name_b)
                continue

            rec = None
            for ch in line:
                rec = builder.feed(ch)
                if rec is not None:
                    break
            if rec is not None:
                records.append(rec)
                srv = rec.server
                server_pts[srv][0 if rec.winner == srv else 1] += 1
                events = score.award_point(rec.winner, fmt)
                print(f"    {rec.describe()}")
                for ev in events:
                    print(f"    {ev}")
                print(f"  {score.line(name_a, name_b)}")
                break

    _summary(records, score, name_a, name_b)


def _predict(server_pts, score, name_a, name_b):
    pa = _rate(server_pts[0], default=0.62)
    pb = _rate(server_pts[1], default=0.62)
    model = MatchModel(pa, pb, fmt=BO3)
    p_a_wins = model.match_prob(score)
    print(f"\n    Observed serve win%: {name_a} {pa:.0%}   {name_b} {pb:.0%}")
    print(f"    P({name_a} wins match) = {p_a_wins:.1%}")
    for (ga, gb), p in model.set_score_dist(score, top=3):
        print(f"      {ga}-{gb} sets: {p:.1%}")
    print()


def _rate(won_lost, default):
    won, lost = won_lost
    total = won + lost
    if total < 4:
        return default
    return won / total


def _summary(records, score, name_a, name_b):
    print()
    print(f"  Final: {score.line(name_a, name_b)}")
    print(f"  Points played: {len(records)}")
    print()


def run_demo():
    print()
    print("  =============================================")
    print("  DEMO: typed keystream, auto-scored")
    print("  =============================================")
    print()

    demo_points = "aw az al. alo alaw alalz aw al. aw az alaw al. aw aw".split()
    score = Score(server=0)
    for tok in demo_points:
        if score.winner is not None:
            break
        b = PointBuilder(score.server)
        rec = None
        for ch in tok:
            rec = b.feed(ch)
        if rec is None:
            continue
        events = score.award_point(rec.winner, BO3)
        print(f"  {tok:<8} {rec.describe():<28} {score.line('A', 'B')}")
        for ev in events:
            print(f"           {ev}")

    print(f"\n  Final: {score.line('A', 'B')}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()

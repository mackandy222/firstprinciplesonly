#!/usr/bin/env python3
"""
First-Principles Football Simulator
====================================
Run a full 90-minute match where every outcome — goals, saves, tackles,
passes — emerges from Newtonian physics, biomechanics, and tactical AI.

Usage:
    python main.py [--seed N]
"""
import sys
import time

from engine.match import Match, Team
from engine.pitch import Pitch, Environment
from data.teams import create_team_a, create_team_b


def main():
    seed = 42
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--seed" and i + 2 < len(sys.argv):
            seed = int(sys.argv[i + 2])

    # --- build teams ---
    players_a, instr_a = create_team_a()
    players_b, instr_b = create_team_b()

    team_a = Team("FC Principles", players_a, instr_a)
    team_b = Team("Athletic First", players_b, instr_b)

    # --- environment ---
    env = Environment()
    env.temperature = 24.0       # warm day
    # env.wind[0] = 2.0          # slight headwind for team attacking right

    # --- simulate ---
    print()
    print(f"  {team_a.name}  vs  {team_b.name}")
    print(f"  Tactics: {_preset_name(instr_a)}  vs  {_preset_name(instr_b)}")
    print(f"  Temp: {env.temperature}C   Wind: {env.wind[0]:.1f} m/s")
    print(f"  {'='*52}")

    match = Match(team_a, team_b, env=env, seed=seed)

    t0 = time.perf_counter()
    events = match.run()
    elapsed = time.perf_counter() - t0

    # --- print events ---
    print()
    for ev in events:
        if ev.type in ("goal", "shot", "injury", "info"):
            print(ev)

    # --- stats ---
    print(match.summary())

    # --- player fitness ---
    print()
    print(f"  {'Player':<16} {'Team':<16} {'Stamina':>8} {'Injured':>8}")
    print(f"  {'-'*50}")
    for p in sorted(match._all_players(),
                    key=lambda x: x.stamina_current):
        tm = match._team_of(p).name
        inj = f"{p.injury_severity:.0%}" if p.injured else "-"
        print(f"  {p.name:<16} {tm:<16} {p.stamina_current:>7.1f} {inj:>8}")

    print(f"\n  Simulated 90 min in {elapsed:.2f}s  (seed={seed})")
    print()


def _preset_name(instr):
    from engine.tactics import PRESETS
    for name, vals in PRESETS.items():
        if all(abs(getattr(instr, k) - v) < 0.01 for k, v in vals.items()):
            return name
    return "custom"


if __name__ == "__main__":
    main()

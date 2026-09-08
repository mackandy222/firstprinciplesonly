#!/usr/bin/env python3
"""
Live Match Predictor
=====================
Feed in real match events as they happen.  The engine infers player
attributes and tactics from the data, then runs Monte Carlo simulations
to predict the rest of the match.

Usage:
    python live_match.py              (interactive mode with demo data)
    python live_match.py --demo       (auto-run a demo match)
"""
import sys
from collections import defaultdict

from engine.inference import PlayerObservations, infer_attributes, infer_tactics
from engine.predictor import predict_from_here
from engine.pitch import Environment


# =====================================================================
# Event parser — turn text commands into observations
# =====================================================================

def parse_event(line, minute, home_team, away_team, registry):
    """
    Parse a single event line and update player observations.

    Supported formats:
        pass  TeamA PlayerA -> PlayerB  [distance]
        shot  TeamA PlayerA  [on_target|off_target|goal]
        tackle TeamA PlayerA wins|loses
        dribble TeamA PlayerA success|fail
        foul  TeamA PlayerA
        corner TeamA
        save  TeamA GKName
        aerial TeamA PlayerA wins|loses
        sprint TeamA PlayerA [speed_ms]
        cross TeamA PlayerA success|fail
        position TeamA PlayerA x_frac y_frac

    Team name is matched to home/away.
    """
    parts = line.strip().split()
    if len(parts) < 3:
        return None

    event_type = parts[0].lower()
    team = parts[1]
    player_name = parts[2]

    obs = _get_or_create(registry, player_name, team)
    obs.touches += 1

    if event_type == "pass":
        obs.passes_attempted += 1
        # look for -> target
        if "->" in parts:
            idx = parts.index("->")
            if idx + 1 < len(parts):
                target_name = parts[idx + 1]
                obs.pass_targets[target_name] += 1
                obs.passes_completed += 1
                target_obs = _get_or_create(registry, target_name, team)
                target_obs.touches += 1

                # distance if provided
                if idx + 2 < len(parts):
                    try:
                        dist = float(parts[idx + 2])
                        obs.pass_distances.append(dist)
                        if dist > 25:
                            obs.long_passes += 1
                        else:
                            obs.short_passes += 1
                    except ValueError:
                        pass
        # direction (simplified: assume forward if not specified)
        obs.forward_passes += 1
        return f"  [{minute}'] PASS: {player_name} -> {parts[parts.index('->') + 1] if '->' in parts else '?'} ({team})"

    elif event_type == "shot":
        obs.shots += 1
        result = parts[3] if len(parts) > 3 else "off_target"
        if result == "goal":
            obs.shots_on_target += 1
            obs.goals += 1
            return f"  [{minute}'] GOAL! {player_name} ({team})"
        elif result == "on_target":
            obs.shots_on_target += 1
            return f"  [{minute}'] SHOT on target: {player_name} ({team})"
        else:
            return f"  [{minute}'] SHOT off target: {player_name} ({team})"

    elif event_type == "tackle":
        obs.tackles_attempted += 1
        if len(parts) > 3 and parts[3] == "wins":
            obs.tackles_won += 1
        return f"  [{minute}'] TACKLE {'won' if len(parts) > 3 and parts[3] == 'wins' else 'lost'}: {player_name} ({team})"

    elif event_type == "dribble":
        obs.dribbles_attempted += 1
        if len(parts) > 3 and parts[3] == "success":
            obs.dribbles_completed += 1
        return None

    elif event_type == "foul":
        obs.fouls_committed += 1
        return f"  [{minute}'] FOUL by {player_name} ({team})"

    elif event_type == "corner":
        obs.corners_taken += 1
        return f"  [{minute}'] CORNER: {team}"

    elif event_type == "save":
        obs.saves += 1
        if obs.position is None:
            obs.position = "GK"
        return f"  [{minute}'] SAVE: {player_name} ({team})"

    elif event_type == "aerial":
        if len(parts) > 3 and parts[3] == "wins":
            obs.aerials_won += 1
        else:
            obs.aerials_lost += 1
        return None

    elif event_type == "sprint":
        obs.sprints += 1
        if len(parts) > 3:
            try:
                obs.sprint_speeds.append(float(parts[3]))
            except ValueError:
                pass
        return None

    elif event_type == "position":
        if len(parts) >= 5:
            try:
                x, y = float(parts[3]), float(parts[4])
                obs.touch_positions.append((x, y))
            except ValueError:
                pass
        return None

    elif event_type == "role":
        if len(parts) > 3:
            obs.position = parts[3]
        return None

    return None


def _get_or_create(registry, name, team):
    key = (team, name)
    if key not in registry:
        registry[key] = PlayerObservations(name, team)
    return registry[key]


# =====================================================================
# Interactive live match
# =====================================================================

def run_interactive():
    print()
    print("  =============================================")
    print("  LIVE MATCH PREDICTOR - First Principles")
    print("  =============================================")
    print()

    home = input("  Home team name: ").strip() or "Home"
    away = input("  Away team name: ").strip() or "Away"

    print(f"\n  {home} vs {away}")
    print(f"  Type events as they happen. Commands:")
    print(f"    pass {home} Rodri -> Bernardo 15")
    print(f"    shot {home} Foden on_target")
    print(f"    shot {away} Salah goal")
    print(f"    tackle {home} Rice wins")
    print(f"    role {home} Ederson GK")
    print(f"    predict            (run prediction)")
    print(f"    stats              (show inferred stats)")
    print(f"    minute N           (set current minute)")
    print(f"    quit")
    print()

    registry = {}
    minute = 0
    home_goals = 0
    away_goals = 0
    corners = {home: 0, away: 0}

    while True:
        try:
            line = input(f"  [{minute}'] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        if line.lower() == "quit":
            break

        if line.lower().startswith("minute"):
            parts = line.split()
            if len(parts) > 1:
                try:
                    minute = int(parts[1])
                except ValueError:
                    pass
            continue

        if line.lower() == "predict":
            _run_prediction(registry, home, away, home_goals, away_goals, minute)
            continue

        if line.lower() == "stats":
            _show_stats(registry, minute)
            continue

        if line.lower() == "tactics":
            _show_tactics(registry, home, away)
            continue

        msg = parse_event(line, minute, home, away, registry)
        if msg:
            print(msg)
            # track goals and corners
            if "GOAL!" in msg:
                if home in msg:
                    home_goals += 1
                else:
                    away_goals += 1
                print(f"  Score: {home} {home_goals}-{away_goals} {away}")
            if "CORNER" in msg:
                if home in line:
                    corners[home] += 1
                else:
                    corners[away] += 1

    print(f"\n  Final: {home} {home_goals}-{away_goals} {away}")
    print(f"  Corners: {home} {corners[home]} - {corners[away]} {away}")


def _run_prediction(registry, home, away, home_goals, away_goals, minute):
    home_obs = [obs for (t, _), obs in registry.items() if t == home]
    away_obs = [obs for (t, _), obs in registry.items() if t == away]

    if len(home_obs) < 3 or len(away_obs) < 3:
        print("  Need more events before predicting (at least 3 players per team)")
        return

    print(f"\n  Running prediction from minute {minute}...")
    print(f"  (inferring stats from {sum(o.touches for o in home_obs + away_obs)} observed events)")

    pred = predict_from_here(
        home_obs, away_obs, home, away,
        current_score=(home_goals, away_goals),
        minute_now=minute,
        n_simulations=20,   # keep low for speed
        seed=None
    )

    print(pred.summary(home, away, (home_goals, away_goals), minute))


def _show_stats(registry, minute):
    print(f"\n  Inferred player attributes at {minute}':")
    print(f"  {'Player':<14} {'Team':<10} {'Pos':<4} {'SPD':>4} {'ACC':>4} "
          f"{'PAS':>4} {'SHT':>4} {'DRB':>4} {'TAC':>4} {'VIS':>4} {'CMP':>4}")
    print(f"  {'-'*70}")

    for (team, name), obs in sorted(registry.items()):
        attrs = infer_attributes(obs, minute)
        pos = obs.position or "?"
        print(f"  {name:<14} {team:<10} {pos:<4} "
              f"{attrs.speed:>4.0f} {attrs.accuracy:>4.0f} "
              f"{attrs.accuracy:>4.0f} {attrs.kick_power:>4.0f} "
              f"{attrs.dribbling:>4.0f} {attrs.tackling:>4.0f} "
              f"{attrs.vision:>4.0f} {attrs.composure:>4.0f}")


def _show_tactics(registry, home, away):
    for team_name in [home, away]:
        all_obs = [obs for (t, _), obs in registry.items() if t == team_name]
        instr, formation = infer_tactics(all_obs, team_name)
        print(f"\n  {team_name} - detected formation: {formation}")
        print(f"    Mentality:  {instr.mentality:.2f}  "
              f"Pressing: {instr.pressing_intensity:.2f}  "
              f"Width: {instr.width:.2f}")
        print(f"    Tempo:      {instr.tempo:.2f}  "
              f"Directness: {instr.directness:.2f}  "
              f"Def.Line: {instr.defensive_line:.2f}")


# =====================================================================
# Demo mode — auto-feed events from a sample match
# =====================================================================

def run_demo():
    print()
    print("  =============================================")
    print("  DEMO: Man City vs Arsenal (simulated events)")
    print("  =============================================")
    print()

    home, away = "City", "Arsenal"
    registry = {}
    home_goals, away_goals = 0, 0

    # pre-set positions
    roles = [
        f"role City Ederson GK", f"role City Walker RB", f"role City Dias CB",
        f"role City Stones CB", f"role City Gvardiol LB", f"role City Rodri CDM",
        f"role City Bernardo CM", f"role City DeBruyne CAM", f"role City Foden LW",
        f"role City Haaland ST", f"role City Saka RW",
        f"role Arsenal Raya GK", f"role Arsenal White RB", f"role Arsenal Saliba CB",
        f"role Arsenal Gabriel CB", f"role Arsenal Zinchenko LB", f"role Arsenal Rice CDM",
        f"role Arsenal Odegaard CAM", f"role Arsenal Havertz CM", f"role Arsenal Saka RW",
        f"role Arsenal Martinelli LW", f"role Arsenal Jesus ST",
    ]
    for r in roles:
        parse_event(r, 0, home, away, registry)

    # simulated first 30 minutes of events
    events = [
        (1, f"pass City Ederson -> Dias 20"),
        (1, f"pass City Dias -> Rodri 15"),
        (2, f"pass City Rodri -> Bernardo 18"),
        (2, f"pass City Bernardo -> DeBruyne 12"),
        (3, f"pass City DeBruyne -> Foden 22"),
        (3, f"dribble City Foden success"),
        (4, f"pass City Foden -> Haaland 15"),
        (5, f"shot City Haaland on_target"),
        (5, f"save Arsenal Raya"),
        (6, f"pass Arsenal Raya -> Saliba 25"),
        (7, f"pass Arsenal Saliba -> Rice 20"),
        (7, f"pass Arsenal Rice -> Odegaard 18"),
        (8, f"pass Arsenal Odegaard -> Saka 25"),
        (8, f"dribble Arsenal Saka success"),
        (9, f"cross Arsenal Saka success"),
        (9, f"aerial City Dias wins"),
        (10, f"pass City Dias -> Rodri 12"),
        (11, f"pass City Rodri -> DeBruyne 20"),
        (12, f"pass City DeBruyne -> Haaland 28"),
        (12, f"dribble City Haaland success"),
        (13, f"shot City Haaland goal"),
        (15, f"pass Arsenal Gabriel -> Rice 18"),
        (15, f"pass Arsenal Rice -> Odegaard 15"),
        (16, f"pass Arsenal Odegaard -> Martinelli 22"),
        (16, f"dribble Arsenal Martinelli success"),
        (17, f"shot Arsenal Martinelli off_target"),
        (18, f"tackle City Rodri wins"),
        (19, f"pass City Rodri -> Bernardo 15"),
        (20, f"pass City Bernardo -> Foden 20"),
        (20, f"pass City Foden -> DeBruyne 10"),
        (21, f"shot City DeBruyne on_target"),
        (21, f"save Arsenal Raya"),
        (23, f"pass Arsenal Raya -> White 22"),
        (23, f"pass Arsenal White -> Odegaard 18"),
        (24, f"pass Arsenal Odegaard -> Saka 20"),
        (24, f"dribble Arsenal Saka success"),
        (25, f"shot Arsenal Saka on_target"),
        (25, f"save City Ederson"),
        (27, f"tackle Arsenal Rice wins"),
        (27, f"pass Arsenal Rice -> Jesus 25"),
        (28, f"shot Arsenal Jesus goal"),
        (30, f"pass City Stones -> Rodri 15"),
        (30, f"pass City Rodri -> DeBruyne 20"),
        (30, f"tackle Arsenal Rice wins"),
    ]

    for minute, event_line in events:
        msg = parse_event(event_line, minute, home, away, registry)
        if msg:
            print(msg)
            if "GOAL!" in msg:
                if home in msg:
                    home_goals += 1
                else:
                    away_goals += 1
                print(f"  Score: {home} {home_goals}-{away_goals} {away}")

    print(f"\n  --- 30 minutes played: {home} {home_goals}-{away_goals} {away} ---")

    # show what the engine inferred
    _show_stats(registry, 30)
    _show_tactics(registry, home, away)

    # run prediction
    print(f"\n  Running 20 simulations of remaining 60 minutes...")
    _run_prediction(registry, home, away, home_goals, away_goals, 30)


# =====================================================================
# main
# =====================================================================

if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()

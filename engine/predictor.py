"""
Prediction engine: given the current match state and inferred player stats,
run N forward simulations to estimate final score, corners, cards, etc.

The key insight: early in the match (5-10 min), predictions are uncertain
because we have little data. By 60-70 min, the inferred stats are solid
and the remaining time is short, so predictions become very accurate.
"""
import numpy as np
from collections import Counter

from engine.match import Match, Team
from engine.pitch import Pitch, Environment
from engine.inference import build_players, infer_tactics
from engine.tactics import ManagerInstructions


class Prediction:
    """Result of a Monte Carlo prediction run."""

    def __init__(self):
        self.simulations = 0
        self.home_goals = []
        self.away_goals = []
        self.home_shots = []
        self.away_shots = []

    @property
    def avg_home_goals(self):
        return np.mean(self.home_goals) if self.home_goals else 0

    @property
    def avg_away_goals(self):
        return np.mean(self.away_goals) if self.away_goals else 0

    @property
    def home_win_pct(self):
        if not self.home_goals:
            return 0
        wins = sum(1 for h, a in zip(self.home_goals, self.away_goals) if h > a)
        return wins / len(self.home_goals) * 100

    @property
    def draw_pct(self):
        if not self.home_goals:
            return 0
        draws = sum(1 for h, a in zip(self.home_goals, self.away_goals) if h == a)
        return draws / len(self.home_goals) * 100

    @property
    def away_win_pct(self):
        if not self.home_goals:
            return 0
        wins = sum(1 for h, a in zip(self.home_goals, self.away_goals) if a > h)
        return wins / len(self.home_goals) * 100

    @property
    def most_likely_score(self):
        if not self.home_goals:
            return (0, 0)
        scores = list(zip(self.home_goals, self.away_goals))
        return Counter(scores).most_common(1)[0][0]

    def score_distribution(self, top_n=5):
        """Most common final scores with probabilities."""
        if not self.home_goals:
            return []
        scores = list(zip(self.home_goals, self.away_goals))
        counts = Counter(scores).most_common(top_n)
        total = len(scores)
        return [(s, c / total * 100) for s, c in counts]

    def summary(self, home_name, away_name, current_score, minute):
        lines = [
            "",
            f"  === PREDICTION at {minute}' ===",
            f"  Current: {home_name} {current_score[0]}-{current_score[1]} {away_name}",
            f"  Simulations: {self.simulations}",
            f"",
            f"  Predicted final (additional goals from now):",
            f"    {home_name}: +{self.avg_home_goals:.1f} goals avg",
            f"    {away_name}: +{self.avg_away_goals:.1f} goals avg",
            f"",
            f"  Match outcome probability:",
            f"    {home_name} win: {self.home_win_pct:.0f}%",
            f"    Draw:            {self.draw_pct:.0f}%",
            f"    {away_name} win: {self.away_win_pct:.0f}%",
            f"",
            f"  Most likely additional scores:",
        ]
        for (h, a), pct in self.score_distribution(5):
            projected_h = current_score[0] + h
            projected_a = current_score[1] + a
            lines.append(f"    {projected_h}-{projected_a}  ({pct:.0f}%)")

        conf = min(100, minute * 1.1)
        lines.append(f"")
        lines.append(f"  Confidence: {conf:.0f}% (based on {minute} min of data)")
        return "\n".join(lines)


def predict_from_here(home_obs, away_obs, home_name, away_name,
                      current_score, minute_now, env=None,
                      n_simulations=50, seed=None):
    """
    Run Monte Carlo forward simulations from the current match state.

    Parameters:
        home_obs: list of PlayerObservations for home team
        away_obs: list of PlayerObservations for away team
        current_score: (home_goals, away_goals)
        minute_now: current match minute (0-90)
        n_simulations: how many forward sims to run
    """
    rng = np.random.RandomState(seed)
    env = env or Environment()
    pitch = Pitch()

    # infer stats from observations so far
    home_players = build_players(home_obs, minute_now)
    away_players = build_players(away_obs, minute_now)

    home_instr, home_formation = infer_tactics(home_obs, home_name)
    away_instr, away_formation = infer_tactics(away_obs, away_name)

    prediction = Prediction()
    prediction.simulations = n_simulations

    # remaining time
    remaining_minutes = max(1, 90 - minute_now)
    # scale the match engine to only simulate the remaining portion
    remaining_seconds = remaining_minutes * 60

    for i in range(n_simulations):
        sim_seed = rng.randint(0, 100000)

        # create fresh copies of players for each simulation
        h_players = build_players(home_obs, minute_now)
        a_players = build_players(away_obs, minute_now)

        # reduce stamina proportional to minutes played
        fatigue_frac = minute_now / 90.0
        for p in h_players + a_players:
            p.stamina_current *= (1.0 - fatigue_frac * 0.6)

        team_h = Team(home_name, h_players, home_instr.copy())
        team_a = Team(away_name, a_players, away_instr.copy())

        match = Match(team_h, team_a, pitch=pitch, env=env, seed=sim_seed)

        # override half duration to only simulate remaining time
        match.HALF_DURATION = remaining_seconds // 2 if minute_now < 45 else remaining_seconds

        # run the partial simulation
        match.run()

        prediction.home_goals.append(team_h.score)
        prediction.away_goals.append(team_a.score)
        prediction.home_shots.append(match.shots[0])
        prediction.away_shots.append(match.shots[1])

    return prediction

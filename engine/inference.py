"""
Inference engine: observe real match events and derive player attributes,
team tactics, and formation shape.

The core idea: we never see a player's "accuracy" stat directly.
We observe that Rodri completed 14/15 passes at avg 18m distance,
and from that we INFER his accuracy is ~88, his vision is ~85, etc.

As more events accumulate, the estimates converge toward reality.
"""
import numpy as np
from collections import defaultdict
from engine.player import Player, PlayerAttributes
from engine.tactics import ManagerInstructions


# =====================================================================
# Event types that can be fed in from a real match
# =====================================================================

class MatchEvent:
    """A single observable event from a real match."""
    def __init__(self, minute, event_type, **data):
        self.minute = minute
        self.type = event_type     # 'pass', 'shot', 'tackle', 'dribble',
                                   # 'interception', 'foul', 'corner',
                                   # 'goal', 'save', 'aerial', 'cross'
        self.data = data

    def __repr__(self):
        return f"[{self.minute}'] {self.type}: {self.data}"


# =====================================================================
# Player stat accumulator — tracks raw observations
# =====================================================================

class PlayerObservations:
    """Raw observations for a single player, updated as events arrive."""

    def __init__(self, name, team, position=None):
        self.name = name
        self.team = team
        self.position = position   # 'GK','CB','LB','RB','CM','LW','RW','ST' etc.

        # passing
        self.passes_attempted = 0
        self.passes_completed = 0
        self.pass_distances = []         # metres
        self.pass_targets = defaultdict(int)   # name -> count (who they pass to)
        self.long_passes = 0
        self.short_passes = 0
        self.forward_passes = 0
        self.backward_passes = 0

        # shooting
        self.shots = 0
        self.shots_on_target = 0
        self.goals = 0
        self.shot_distances = []

        # defending
        self.tackles_attempted = 0
        self.tackles_won = 0
        self.interceptions = 0
        self.aerials_won = 0
        self.aerials_lost = 0
        self.fouls_committed = 0

        # dribbling
        self.dribbles_attempted = 0
        self.dribbles_completed = 0

        # physical (if available from tracking data)
        self.sprint_speeds = []          # m/s observations
        self.distance_covered = 0.0      # metres
        self.sprints = 0

        # crosses / set pieces
        self.crosses_attempted = 0
        self.crosses_completed = 0
        self.corners_taken = 0

        # goalkeeping
        self.saves = 0
        self.goals_conceded = 0
        self.distribution_attempts = 0

        # touches
        self.touches = 0
        self.touch_positions = []        # (x_frac, y_frac) on pitch

    @property
    def pass_completion_rate(self):
        if self.passes_attempted == 0:
            return 0.5
        return self.passes_completed / self.passes_attempted

    @property
    def tackle_success_rate(self):
        if self.tackles_attempted == 0:
            return 0.5
        return self.tackles_won / self.tackles_attempted

    @property
    def shot_accuracy(self):
        if self.shots == 0:
            return 0.5
        return self.shots_on_target / self.shots

    @property
    def avg_pass_distance(self):
        if not self.pass_distances:
            return 15.0
        return np.mean(self.pass_distances)

    @property
    def avg_position(self):
        """Average position on pitch (x_frac, y_frac)."""
        if not self.touch_positions:
            return (0.5, 0.5)
        arr = np.array(self.touch_positions)
        return (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))


# =====================================================================
# Stat inference — convert observations into engine-compatible attributes
# =====================================================================

def infer_attributes(obs, minutes_elapsed):
    """
    Convert raw observations into PlayerAttributes.

    With few events (early in match), we lean on position-based priors.
    As events accumulate, the observed data dominates.
    """
    # confidence: how much we trust observations vs priors (0 to 1)
    event_count = (obs.passes_attempted + obs.shots + obs.tackles_attempted
                   + obs.dribbles_attempted + obs.touches)
    confidence = min(1.0, event_count / 40.0)   # full confidence ~40 events

    # position-based priors (what we'd expect before seeing any data)
    prior = _position_prior(obs.position or "CM")

    def blend(observed_val, prior_val):
        return observed_val * confidence + prior_val * (1.0 - confidence)

    # --- technical ---
    pass_acc = obs.pass_completion_rate
    accuracy = blend(pass_acc * 100, prior.get("accuracy", 70))

    avg_dist = obs.avg_pass_distance
    kick_power = blend(min(95, 50 + avg_dist * 1.5), prior.get("kick_power", 70))

    vision = blend(
        min(95, 50 + len(set(obs.pass_targets.keys())) * 5  # variety of targets
            + obs.forward_passes * 0.5),
        prior.get("vision", 70)
    )

    first_touch = blend(
        min(95, 55 + obs.pass_completion_rate * 30
            + (obs.dribbles_completed / max(1, obs.dribbles_attempted)) * 15),
        prior.get("first_touch", 70)
    )

    dribbling = blend(
        min(95, 50 + (obs.dribbles_completed / max(1, obs.dribbles_attempted)) * 40
            + obs.dribbles_completed * 2),
        prior.get("dribbling", 70)
    )

    tackling = blend(
        min(95, 45 + obs.tackle_success_rate * 40 + obs.interceptions * 3),
        prior.get("tackling", 70)
    )

    heading = blend(
        min(95, 50 + (obs.aerials_won / max(1, obs.aerials_won + obs.aerials_lost)) * 40),
        prior.get("heading", 70)
    )

    composure = blend(
        min(95, 55 + obs.shot_accuracy * 25
            + obs.pass_completion_rate * 15
            - obs.fouls_committed * 3),
        prior.get("composure", 70)
    )

    # --- physical ---
    speed = prior.get("speed", 70)
    if obs.sprint_speeds:
        # map real sprint speed to 0-100 (6 m/s = 40, 10 m/s = 95)
        max_sprint = max(obs.sprint_speeds)
        speed = blend(min(95, max(40, 40 + (max_sprint - 6) * 13.75)), speed)

    acceleration = blend(speed * 0.95, prior.get("acceleration", 70))

    stamina_est = prior.get("stamina", 70)
    if minutes_elapsed > 30 and obs.distance_covered > 0:
        # high distance covered = good stamina
        km_per_90 = obs.distance_covered / 1000 * (90 / max(1, minutes_elapsed))
        stamina_est = blend(min(95, 40 + km_per_90 * 5), stamina_est)

    strength = prior.get("strength", 70)
    if obs.aerials_won + obs.aerials_lost > 0:
        aerial_rate = obs.aerials_won / (obs.aerials_won + obs.aerials_lost)
        strength = blend(min(95, 50 + aerial_rate * 35 + obs.tackles_won * 2), strength)

    # --- mental ---
    positioning = blend(
        min(95, 55 + obs.interceptions * 4
            + obs.touches * 0.3),
        prior.get("positioning", 70)
    )

    work_rate = blend(
        min(95, 50 + obs.sprints * 2 + obs.tackles_attempted * 2),
        prior.get("work_rate", 70)
    )

    anticipation = blend(
        min(95, 50 + obs.interceptions * 5 + obs.tackles_won * 2),
        prior.get("anticipation", 70)
    )

    # --- goalkeeper ---
    reflexes = prior.get("reflexes", 10)
    diving = prior.get("diving", 10)
    handling = prior.get("handling", 10)
    if obs.position == "GK":
        if obs.saves + obs.goals_conceded > 0:
            save_rate = obs.saves / (obs.saves + obs.goals_conceded)
            reflexes = blend(min(95, 50 + save_rate * 40), 78)
            diving = blend(min(95, 50 + save_rate * 35), 76)
            handling = blend(min(95, 50 + save_rate * 30), 74)

    return PlayerAttributes(
        speed=speed, acceleration=acceleration, stamina=stamina_est,
        strength=strength, agility=blend(70, prior.get("agility", 70)),
        kick_power=kick_power, accuracy=accuracy, first_touch=first_touch,
        dribbling=dribbling, tackling=tackling, heading=heading,
        vision=vision, composure=composure, positioning=positioning,
        work_rate=work_rate, anticipation=anticipation,
        reflexes=reflexes, diving=diving, handling=handling,
        height_cm=prior.get("height_cm", 180),
        weight_kg=prior.get("weight_kg", 76),
    )


# =====================================================================
# Tactical inference — detect formation and style from passing patterns
# =====================================================================

def infer_tactics(team_observations, team_name):
    """
    Derive ManagerInstructions from observed passing patterns and positions.

    Returns (ManagerInstructions, formation_description).
    """
    players = [obs for obs in team_observations if obs.team == team_name]
    instr = ManagerInstructions()

    if not players:
        return instr, "unknown"

    total_passes = sum(p.passes_attempted for p in players)
    total_forward = sum(p.forward_passes for p in players)
    total_backward = sum(p.backward_passes for p in players)
    total_long = sum(p.long_passes for p in players)
    total_short = sum(p.short_passes for p in players)
    total_shots = sum(p.shots for p in players)
    avg_pass_rate = np.mean([p.pass_completion_rate for p in players])
    total_tackles = sum(p.tackles_attempted for p in players)

    # --- directness ---
    if total_passes > 0:
        fwd_ratio = total_forward / total_passes
        long_ratio = total_long / total_passes
        instr.directness = np.clip(fwd_ratio * 0.5 + long_ratio * 0.8, 0.05, 0.95)
    else:
        instr.directness = 0.5

    # --- tempo ---
    # high pass count per minute = high tempo
    minutes = max(1, max(p.touches for p in players) * 0.1)  # rough estimate
    passes_per_min = total_passes / max(1, minutes)
    instr.tempo = np.clip(passes_per_min / 15.0, 0.1, 0.95)

    # --- pressing ---
    tackles_per_player = total_tackles / max(1, len(players))
    instr.pressing_intensity = np.clip(tackles_per_player / 3.0, 0.1, 0.95)

    # --- mentality ---
    shot_ratio = total_shots / max(1, total_passes) * 50
    instr.mentality = np.clip(0.3 + shot_ratio + instr.directness * 0.2, 0.1, 0.95)

    # --- width ---
    # if wingers/fullbacks have lots of touches, team plays wide
    wide_players = [p for p in players
                    if p.position in ("LW", "RW", "LM", "RM", "LB", "RB")]
    if wide_players and total_passes > 0:
        wide_pass_share = sum(p.passes_attempted for p in wide_players) / total_passes
        instr.width = np.clip(wide_pass_share * 2.5, 0.2, 0.95)

    # --- defensive line ---
    # infer from average position of defenders
    defenders = [p for p in players if p.position in ("CB", "LB", "RB")]
    if defenders:
        avg_def_x = np.mean([p.avg_position[0] for p in defenders])
        instr.defensive_line = np.clip(avg_def_x * 1.5, 0.1, 0.9)

    # --- detect formation shape ---
    formation = _detect_formation(players)

    return instr, formation


def _detect_formation(players):
    """Guess formation from average positions (e.g. '4-3-3')."""
    if len(players) < 10:
        return "unknown"

    # count players by rough band
    positions = {p.name: p.avg_position for p in players if p.position != "GK"}
    if not positions:
        return "unknown"

    xs = [pos[0] for pos in positions.values()]
    if not xs:
        return "unknown"

    # split into bands: defense (0-0.35), midfield (0.35-0.6), attack (0.6-1.0)
    defense = sum(1 for x in xs if x < 0.35)
    midfield = sum(1 for x in xs if 0.35 <= x < 0.60)
    attack = sum(1 for x in xs if x >= 0.60)

    return f"{defense}-{midfield}-{attack}"


# =====================================================================
# Build simulation-ready players from observations
# =====================================================================

def build_players(observations, minutes_elapsed):
    """Convert a list of PlayerObservations into simulation Player objects."""
    players = []
    for obs in observations:
        attrs = infer_attributes(obs, minutes_elapsed)
        # determine formation slot from average position
        ax, ay = obs.avg_position
        slot = (np.clip(ax, 0.04, 0.70), np.clip(ay, 0.08, 0.92))
        p = Player(obs.name, obs.position or "CM", slot, attrs)
        players.append(p)
    return players


# =====================================================================
# Position priors — reasonable defaults before we have data
# =====================================================================

def _position_prior(role):
    """Return a dict of expected attribute values for a given position."""
    _PRIORS = {
        "GK":  dict(speed=50, acceleration=50, stamina=65, strength=72,
                     agility=62, kick_power=70, accuracy=55, first_touch=50,
                     dribbling=25, tackling=15, heading=30, vision=50,
                     composure=72, positioning=50, work_rate=50,
                     anticipation=58, reflexes=80, diving=78, handling=75,
                     height_cm=188, weight_kg=84),
        "CB":  dict(speed=62, acceleration=58, stamina=75, strength=82,
                     agility=55, kick_power=62, accuracy=58, first_touch=58,
                     dribbling=45, tackling=82, heading=80, vision=55,
                     composure=75, positioning=82, work_rate=72,
                     anticipation=80, height_cm=186, weight_kg=82),
        "LB":  dict(speed=78, acceleration=75, stamina=82, strength=68,
                     agility=74, kick_power=60, accuracy=65, first_touch=65,
                     dribbling=65, tackling=72, heading=55, vision=62,
                     composure=68, positioning=72, work_rate=85,
                     anticipation=72, height_cm=176, weight_kg=72),
        "RB":  dict(speed=78, acceleration=75, stamina=82, strength=68,
                     agility=74, kick_power=60, accuracy=65, first_touch=65,
                     dribbling=65, tackling=72, heading=55, vision=62,
                     composure=68, positioning=72, work_rate=85,
                     anticipation=72, height_cm=178, weight_kg=74),
        "CDM": dict(speed=65, acceleration=62, stamina=85, strength=80,
                     agility=65, kick_power=68, accuracy=65, first_touch=70,
                     dribbling=60, tackling=85, heading=72, vision=72,
                     composure=78, positioning=82, work_rate=90,
                     anticipation=82, height_cm=182, weight_kg=78),
        "CM":  dict(speed=68, acceleration=65, stamina=80, strength=72,
                     agility=70, kick_power=68, accuracy=72, first_touch=75,
                     dribbling=70, tackling=68, heading=60, vision=78,
                     composure=75, positioning=76, work_rate=78,
                     anticipation=75, height_cm=180, weight_kg=76),
        "CAM": dict(speed=72, acceleration=72, stamina=72, strength=60,
                     agility=80, kick_power=72, accuracy=78, first_touch=82,
                     dribbling=80, tackling=40, heading=50, vision=85,
                     composure=78, positioning=72, work_rate=65,
                     anticipation=78, height_cm=176, weight_kg=70),
        "LM":  dict(speed=82, acceleration=80, stamina=78, strength=60,
                     agility=82, kick_power=68, accuracy=72, first_touch=78,
                     dribbling=82, tackling=42, heading=48, vision=72,
                     composure=72, positioning=68, work_rate=72,
                     anticipation=70, height_cm=174, weight_kg=68),
        "RM":  dict(speed=82, acceleration=80, stamina=78, strength=60,
                     agility=82, kick_power=68, accuracy=72, first_touch=78,
                     dribbling=82, tackling=42, heading=48, vision=72,
                     composure=72, positioning=68, work_rate=72,
                     anticipation=70, height_cm=175, weight_kg=69),
        "LW":  dict(speed=85, acceleration=82, stamina=75, strength=60,
                     agility=85, kick_power=72, accuracy=75, first_touch=80,
                     dribbling=85, tackling=32, heading=48, vision=72,
                     composure=74, positioning=68, work_rate=65,
                     anticipation=70, height_cm=176, weight_kg=70),
        "RW":  dict(speed=84, acceleration=82, stamina=76, strength=62,
                     agility=84, kick_power=74, accuracy=76, first_touch=80,
                     dribbling=84, tackling=34, heading=50, vision=72,
                     composure=74, positioning=70, work_rate=68,
                     anticipation=72, height_cm=177, weight_kg=71),
        "ST":  dict(speed=78, acceleration=76, stamina=72, strength=78,
                     agility=75, kick_power=82, accuracy=80, first_touch=78,
                     dribbling=72, tackling=28, heading=78, vision=65,
                     composure=82, positioning=85, work_rate=60,
                     anticipation=82, height_cm=183, weight_kg=79),
        "CF":  dict(speed=74, acceleration=72, stamina=72, strength=72,
                     agility=78, kick_power=80, accuracy=80, first_touch=82,
                     dribbling=78, tackling=30, heading=72, vision=75,
                     composure=82, positioning=82, work_rate=65,
                     anticipation=80, height_cm=181, weight_kg=77),
    }
    return _PRIORS.get(role, _PRIORS["CM"])

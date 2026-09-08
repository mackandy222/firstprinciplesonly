"""
Two default teams with distinct styles, realistic sizes, and varied attributes.
Every attribute feeds the first-principles engine — no cosmetic stats.
"""
from engine.player import Player, PlayerAttributes
from engine.tactics import ManagerInstructions, apply_preset


# ── role templates (sane defaults, overridden per-player) ────────────

_TEMPLATES = {
    "GK": dict(speed=50, acceleration=50, stamina=65, strength=72, agility=62,
               kick_power=70, accuracy=55, first_touch=50, dribbling=25,
               tackling=15, heading=30, vision=50, composure=72, positioning=50,
               work_rate=50, anticipation=58, reflexes=80, diving=78, handling=75,
               height_cm=188, weight_kg=84),
    "CB": dict(speed=62, acceleration=58, stamina=75, strength=82, agility=55,
               kick_power=62, accuracy=58, first_touch=58, dribbling=45,
               tackling=82, heading=80, vision=55, composure=75, positioning=82,
               work_rate=72, anticipation=80, height_cm=186, weight_kg=82),
    "LB": dict(speed=78, acceleration=75, stamina=82, strength=68, agility=74,
               kick_power=60, accuracy=65, first_touch=65, dribbling=65,
               tackling=72, heading=55, vision=62, composure=68, positioning=72,
               work_rate=85, anticipation=72, height_cm=176, weight_kg=72),
    "RB": dict(speed=78, acceleration=75, stamina=82, strength=68, agility=74,
               kick_power=60, accuracy=65, first_touch=65, dribbling=65,
               tackling=72, heading=55, vision=62, composure=68, positioning=72,
               work_rate=85, anticipation=72, height_cm=178, weight_kg=74),
    "CM": dict(speed=68, acceleration=65, stamina=80, strength=72, agility=70,
               kick_power=68, accuracy=72, first_touch=75, dribbling=70,
               tackling=68, heading=60, vision=78, composure=75, positioning=76,
               work_rate=78, anticipation=75, height_cm=180, weight_kg=76),
    "CAM": dict(speed=72, acceleration=72, stamina=72, strength=60, agility=80,
                kick_power=72, accuracy=78, first_touch=82, dribbling=80,
                tackling=40, heading=50, vision=85, composure=78, positioning=72,
                work_rate=65, anticipation=78, height_cm=176, weight_kg=70),
    "LM": dict(speed=82, acceleration=80, stamina=78, strength=60, agility=82,
               kick_power=68, accuracy=72, first_touch=78, dribbling=82,
               tackling=42, heading=48, vision=72, composure=72, positioning=68,
               work_rate=72, anticipation=70, height_cm=174, weight_kg=68),
    "RM": dict(speed=82, acceleration=80, stamina=78, strength=60, agility=82,
               kick_power=68, accuracy=72, first_touch=78, dribbling=82,
               tackling=42, heading=48, vision=72, composure=72, positioning=68,
               work_rate=72, anticipation=70, height_cm=175, weight_kg=69),
    "LW": dict(speed=88, acceleration=85, stamina=75, strength=58, agility=88,
               kick_power=72, accuracy=75, first_touch=82, dribbling=88,
               tackling=30, heading=45, vision=75, composure=75, positioning=68,
               work_rate=65, anticipation=72, height_cm=175, weight_kg=68),
    "RW": dict(speed=86, acceleration=84, stamina=78, strength=62, agility=85,
               kick_power=75, accuracy=78, first_touch=80, dribbling=85,
               tackling=32, heading=52, vision=72, composure=76, positioning=70,
               work_rate=70, anticipation=74, height_cm=177, weight_kg=71),
    "ST": dict(speed=78, acceleration=76, stamina=72, strength=78, agility=75,
               kick_power=85, accuracy=82, first_touch=78, dribbling=72,
               tackling=28, heading=78, vision=65, composure=82, positioning=85,
               work_rate=60, anticipation=82, height_cm=185, weight_kg=80),
    "CDM": dict(speed=65, acceleration=62, stamina=85, strength=80, agility=65,
                kick_power=68, accuracy=65, first_touch=70, dribbling=60,
                tackling=85, heading=72, vision=72, composure=78, positioning=82,
                work_rate=90, anticipation=82, height_cm=182, weight_kg=78),
    "CF": dict(speed=74, acceleration=72, stamina=72, strength=72, agility=78,
               kick_power=80, accuracy=80, first_touch=82, dribbling=78,
               tackling=30, heading=72, vision=75, composure=82, positioning=82,
               work_rate=65, anticipation=80, height_cm=181, weight_kg=77),
}


def _p(name, role, slot, **overrides):
    base = _TEMPLATES.get(role, {}).copy()
    base.update(overrides)
    return Player(name, role, slot, PlayerAttributes(**base))


# =====================================================================
# Team A — FC Principles   4-3-3   (possession / tiki-taka)
# =====================================================================

def create_team_a():
    players = [
        _p("Martinez",    "GK",  (0.04, 0.50), reflexes=86, diving=84, handling=82,
           height_cm=190, weight_kg=86),
        _p("Robertson",   "LB",  (0.20, 0.12), speed=82, stamina=88, work_rate=90,
           height_cm=178, weight_kg=72),
        _p("Van Dijk",    "CB",  (0.17, 0.37), strength=90, tackling=88, heading=90,
           composure=85, height_cm=193, weight_kg=92),
        _p("Ramos",       "CB",  (0.17, 0.63), strength=85, tackling=85, heading=88,
           aggression=82, height_cm=184, weight_kg=82),
        _p("Alexander",   "RB",  (0.20, 0.88), speed=80, accuracy=72, dribbling=72,
           height_cm=175, weight_kg=70),
        _p("Modric",      "CM",  (0.38, 0.25), vision=92, accuracy=88, first_touch=90,
           composure=88, dribbling=82, height_cm=172, weight_kg=66),
        _p("De Bruyne",   "CM",  (0.35, 0.50), vision=94, accuracy=90, kick_power=85,
           first_touch=88, composure=85, height_cm=181, weight_kg=76),
        _p("Kroos",       "CM",  (0.38, 0.75), vision=90, accuracy=92, composure=90,
           positioning=82, tempo=85, height_cm=183, weight_kg=78),
        _p("Neymar",      "LW",  (0.62, 0.12), dribbling=94, agility=92, speed=85,
           accuracy=80, composure=78, height_cm=175, weight_kg=68),
        _p("Lewandowski", "ST",  (0.68, 0.50), kick_power=90, accuracy=88,
           heading=88, composure=90, positioning=92,
           height_cm=185, weight_kg=80),
        _p("Salah",       "RW",  (0.62, 0.88), speed=90, acceleration=88,
           dribbling=88, accuracy=82, composure=80,
           height_cm=175, weight_kg=71),
    ]
    instr = ManagerInstructions()
    apply_preset(instr, "tiki_taka")
    return players, instr


# =====================================================================
# Team B — Athletic First   4-4-2   (counter-attack / physical)
# =====================================================================

def create_team_b():
    players = [
        _p("Neuer",     "GK",  (0.04, 0.50), reflexes=88, diving=86, handling=84,
           speed=60, kick_power=80, height_cm=193, weight_kg=92),
        _p("Davies",    "LB",  (0.20, 0.12), speed=92, acceleration=90, stamina=85,
           height_cm=183, weight_kg=75),
        _p("Koulibaly", "CB",  (0.17, 0.37), strength=92, tackling=86, heading=85,
           speed=68, height_cm=187, weight_kg=89),
        _p("Chiellini", "CB",  (0.17, 0.63), strength=88, tackling=88, anticipation=90,
           composure=85, height_cm=187, weight_kg=85),
        _p("Hakimi",    "RB",  (0.20, 0.88), speed=90, acceleration=88, stamina=82,
           dribbling=75, height_cm=181, weight_kg=73),
        _p("Mane",      "LM",  (0.42, 0.12), speed=88, acceleration=86,
           dribbling=85, kick_power=78, work_rate=85,
           height_cm=175, weight_kg=69),
        _p("Kante",     "CM",  (0.38, 0.37), tackling=90, stamina=95, work_rate=95,
           anticipation=88, speed=76, strength=72,
           height_cm=168, weight_kg=68),
        _p("Casemiro",  "CDM", (0.38, 0.63), tackling=88, strength=86,
           positioning=85, heading=78, stamina=82,
           height_cm=185, weight_kg=84),
        _p("Di Maria",  "RM",  (0.42, 0.88), speed=85, dribbling=86, accuracy=82,
           vision=82, agility=85, height_cm=180, weight_kg=75),
        _p("Mbappe",    "ST",  (0.62, 0.37), speed=96, acceleration=95,
           dribbling=88, kick_power=82, composure=80,
           height_cm=178, weight_kg=73),
        _p("Haaland",   "ST",  (0.62, 0.63), kick_power=92, strength=88,
           heading=85, speed=82, acceleration=80, composure=82,
           height_cm=194, weight_kg=88),
    ]
    instr = ManagerInstructions()
    apply_preset(instr, "counter_attack")
    return players, instr

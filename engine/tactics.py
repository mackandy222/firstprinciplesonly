"""
Tactical AI — every decision is shaped by manager instructions,
player attributes, and emergent game state.  Outcomes are never
dice-rolled: the AI picks targets and the physics engine resolves them.
"""
import numpy as np


# =====================================================================
# Manager Instructions  (set per-team, changeable mid-match)
# =====================================================================

class ManagerInstructions:
    """All values 0-1.  Presets provide sensible defaults."""

    def __init__(self):
        self.mentality          = 0.50   # 0=ultra-defensive  1=all-out-attack
        self.pressing_intensity = 0.50   # how aggressively team presses
        self.defensive_line     = 0.50   # 0=deep-block  1=high-line
        self.width              = 0.50   # 0=narrow  1=very wide
        self.tempo              = 0.50   # 0=slow build-up  1=fast transitions
        self.directness         = 0.50   # 0=short/tiki-taka  1=long-ball/route-one

    def copy(self):
        m = ManagerInstructions()
        for k, v in self.__dict__.items():
            setattr(m, k, v)
        return m

PRESETS = {
    "balanced": dict(mentality=0.50, pressing_intensity=0.50,
                     defensive_line=0.50, width=0.50, tempo=0.50, directness=0.50),
    "park_the_bus": dict(mentality=0.15, pressing_intensity=0.15,
                         defensive_line=0.15, width=0.30, tempo=0.30, directness=0.80),
    "possession": dict(mentality=0.50, pressing_intensity=0.65,
                        defensive_line=0.60, width=0.60, tempo=0.35, directness=0.15),
    "tiki_taka": dict(mentality=0.55, pressing_intensity=0.85,
                       defensive_line=0.70, width=0.50, tempo=0.40, directness=0.10),
    "counter_attack": dict(mentality=0.35, pressing_intensity=0.30,
                            defensive_line=0.25, width=0.50, tempo=0.85, directness=0.70),
    "wing_play": dict(mentality=0.60, pressing_intensity=0.50,
                       defensive_line=0.50, width=0.90, tempo=0.60, directness=0.50),
    "route_one": dict(mentality=0.55, pressing_intensity=0.45,
                       defensive_line=0.40, width=0.55, tempo=0.75, directness=0.95),
    "all_out_attack": dict(mentality=0.90, pressing_intensity=0.80,
                            defensive_line=0.80, width=0.70, tempo=0.70, directness=0.50),
    "gegenpressing": dict(mentality=0.65, pressing_intensity=0.95,
                           defensive_line=0.75, width=0.60, tempo=0.80, directness=0.45),
}

def apply_preset(instructions, name):
    for k, v in PRESETS[name].items():
        setattr(instructions, k, v)


# =====================================================================
# Chemistry helpers
# =====================================================================

def pair_chemistry(a, b):
    """Simple chemistry score (0-1) between two players."""
    # complementary work-rates and compatible positioning sense
    wr_diff = abs(a.attrs.work_rate - b.attrs.work_rate)
    pos_avg = (a.attrs.positioning + b.attrs.positioning) / 2.0
    vis_avg = (a.attrs.vision + b.attrs.vision) / 2.0
    return np.clip((pos_avg + vis_avg - wr_diff) / 200.0, 0.0, 1.0)


# =====================================================================
# Execution quality — how well a player enacts the manager's plan
# =====================================================================

def _execution_factor(player):
    """0-1: how cleanly this player can carry out instructions."""
    mental = (player.attrs.composure + player.attrs.positioning
              + player.attrs.vision + player.attrs.anticipation) / 4.0
    fatigue = player.stamina_current / max(1, player.attrs.stamina)
    injury  = 1.0 - player.injury_severity * 0.4
    return np.clip(mental / 100.0 * fatigue * injury, 0.1, 1.0)


# =====================================================================
# Main entry point — called once per decision tick for each player
# =====================================================================

def get_decision(player, ball, own_team, opp_team, pitch, atk_dir,
                 instructions, rng):
    """
    Returns (target_pos_2d, sprint_bool, ball_action | None).
    ball_action: ('shoot', target, power, loft)
               | ('pass',  target, power, loft)
               | ('tackle',)
               | None
    """
    if player.role == "GK":
        return _gk_decision(player, ball, own_team, opp_team,
                            pitch, atk_dir, instructions, rng)

    if player.has_ball:
        return _on_ball(player, ball, own_team, opp_team,
                        pitch, atk_dir, instructions, rng)

    team_has_ball = (ball.carrier is not None
                     and ball.carrier.team_idx == player.team_idx)
    if team_has_ball:
        return _off_ball_attack(player, ball, own_team, opp_team,
                                pitch, atk_dir, instructions, rng)
    return _defending(player, ball, own_team, opp_team,
                      pitch, atk_dir, instructions, rng)


# =====================================================================
# On-ball decisions
# =====================================================================

def _on_ball(player, ball, own_team, opp_team, pitch, atk_dir, instr, rng):
    goal = pitch.goal_center(1 if atk_dir == 1 else 0)
    dist_to_goal = np.linalg.norm(goal - player.pos)
    exe = _execution_factor(player)
    pressure = _get_pressure(player, opp_team)

    # --- shoot score ---
    shoot_score = 0.0
    # only forwards/midfielders should look to shoot from range;
    # defenders only shoot if very close
    max_range = 25.0 if player.role in ("ST","CF","CAM","LW","RW") else 14.0
    if dist_to_goal < max_range:
        angle   = _goal_angle(player.pos, pitch, atk_dir)
        blocked = _shot_blocked(player.pos, goal, opp_team)
        shoot_score = ((1.0 - dist_to_goal / max_range) ** 1.5 * angle
                       * (0.2 if blocked else 1.0)
                       * player.attrs.composure / 100.0)
        # mentality: attacking teams shoot more
        shoot_score *= 0.4 + instr.mentality * 0.5

    # --- pass score ---
    best_target = None
    best_pass_score = 0.0
    for tm in own_team:
        if tm is player or tm.role == "GK":
            continue
        ps = _evaluate_pass(player, tm, opp_team, pitch, atk_dir, instr)
        if ps > best_pass_score:
            best_pass_score = ps
            best_target = tm

    # --- dribble score ---
    dribble_score = _evaluate_dribble(player, opp_team, atk_dir, pitch, instr)

    # execution noise — weaker players make worse choices
    noise = (1.0 - exe) * 0.2
    shoot_score   += rng.normal(0, 0.08 + noise)
    best_pass_score += rng.normal(0, 0.08 + noise)
    dribble_score += rng.normal(0, 0.08 + noise)

    # under pressure → prefer passing (unless park-the-bus clearance)
    if pressure > 0.7:
        best_pass_score += 0.3
        dribble_score   -= 0.2
        if instr.directness > 0.7:
            best_pass_score += 0.15  # boot it

    # --- pick action ---
    if shoot_score > best_pass_score and shoot_score > dribble_score and shoot_score > 0.50:
        target = _aim_shot(player, pitch, atk_dir, rng)
        power  = min(1.0, 0.7 + dist_to_goal / 50.0)
        loft   = 0.05 if dist_to_goal < 15 else 0.15
        return player.pos.copy(), False, ("shoot", target, power, loft)

    if best_target is not None and best_pass_score > dribble_score:
        target = _lead_pass(player, best_target, pitch, atk_dir)
        dist   = np.linalg.norm(target - player.pos)
        # directness: high → lofted long balls; low → short ground passes
        power  = np.clip(dist / 40.0, 0.2, 0.9)
        loft   = instr.directness * 0.4 if dist > 25 else instr.directness * 0.1
        return player.pos.copy(), False, ("pass", target, power, loft)

    # dribble
    target = _dribble_target(player, opp_team, pitch, atk_dir)
    return target, True, None


# =====================================================================
# Off-ball attack
# =====================================================================

def _off_ball_attack(player, ball, own_team, opp_team, pitch, atk_dir,
                     instr, rng):
    base = player.formation_pos(pitch.length, pitch.width, atk_dir,
                                ball.pos[0], ball.pos[1])

    # adjust formation depth by mentality
    goal = pitch.goal_center(1 if atk_dir == 1 else 0)
    mentality_push = (instr.mentality - 0.5) * 15.0      # ±7.5 m
    if atk_dir == 1:
        base[0] += mentality_push
    else:
        base[0] -= mentality_push

    # adjust width
    center_y = pitch.width / 2.0
    base[1] = center_y + (base[1] - center_y) * (0.6 + instr.width * 0.8)

    exe = _execution_factor(player)

    if player.role in ("ST", "LW", "RW", "CAM", "CF"):
        # attacking runs — quality depends on anticipation + execution
        run_aggression = 0.2 + exe * 0.3 + instr.mentality * 0.3
        run_target = base + (goal - base) * run_aggression * 0.3
        # find space away from defenders
        for opp in opp_team:
            if np.linalg.norm(opp.pos - run_target) < 5:
                perp = np.array([-(goal[1] - player.pos[1]),
                                  goal[0] - player.pos[0]])
                pn = np.linalg.norm(perp)
                if pn > 0.01:
                    run_target += (perp / pn) * 3 * rng.choice([-1, 1])
                break
        target = pitch.clamp(run_target)
    else:
        target = pitch.clamp(base)

    sprint = np.linalg.norm(target - player.pos) > 8
    return target, sprint, None


# =====================================================================
# Defending
# =====================================================================

def _defending(player, ball, own_team, opp_team, pitch, atk_dir,
               instr, rng):
    ball_pos = ball.xy
    own_goal = pitch.goal_center(0 if atk_dir == 1 else 1)

    base = player.formation_pos(pitch.length, pitch.width, atk_dir,
                                ball.pos[0], ball.pos[1])

    # defensive line depth
    line_shift = (instr.defensive_line - 0.5) * 20.0    # ±10 m
    if atk_dir == 1:
        base[0] += line_shift
    else:
        base[0] -= line_shift

    # am I closest outfield player to ball?
    my_dist = np.linalg.norm(player.pos - ball_pos)
    closest = True
    for tm in own_team:
        if tm is player or tm.role == "GK":
            continue
        if np.linalg.norm(tm.pos - ball_pos) < my_dist - 1.0:
            closest = False
            break

    # pressing decision shaped by tactical intensity
    press_range = 8.0 + instr.pressing_intensity * 20.0   # 8-28 m
    if closest and my_dist < press_range:
        target = ball_pos.copy()
        sprint = True
    else:
        # hold shape, shift toward ball
        to_ball = ball_pos - base
        target = base + to_ball * 0.2
        target = pitch.clamp(target)
        sprint = np.linalg.norm(target - player.pos) > 15

    # tackle attempt
    action = None
    if ball.carrier is not None and ball.carrier.team_idx != player.team_idx:
        if my_dist < player.TACKLE_RADIUS:
            action = ("tackle",)

    return target, sprint, action


# =====================================================================
# Goalkeeper
# =====================================================================

def _gk_decision(player, ball, own_team, opp_team, pitch, atk_dir,
                 instr, rng):
    own_goal_side = 0 if atk_dir == 1 else 1
    goal_center   = pitch.goal_center(own_goal_side)
    goal_x        = 0.0 if own_goal_side == 0 else pitch.length

    # --- distribution ---
    if player.has_ball:
        best, best_d = None, float("inf")
        for tm in own_team:
            if tm is player:
                continue
            if tm.role in ("CB", "LB", "RB", "CDM", "CM"):
                d = np.linalg.norm(tm.pos - player.pos)
                if d < best_d:
                    best_d = d
                    best = tm
        if best is not None:
            power = np.clip(best_d / 50, 0.3, 0.7)
            loft  = 0.15 if best_d > 30 else 0.05
            if instr.directness > 0.7:
                # route-one: boot it long
                power = 0.9
                loft  = 0.5
            return player.pos.copy(), False, ("pass", best.pos.copy(), power, loft)
        return player.pos.copy(), False, None

    # --- positioning ---
    btg = goal_center - ball.xy
    btg_d = np.linalg.norm(btg)
    if btg_d > 0.1:
        off = 5.0 + min(10.0, btg_d * 0.1)
        target = goal_center - (btg / btg_d) * off
    else:
        target = goal_center.copy()

    gk_x = np.clip(target[0], goal_x - 16, goal_x + 16)
    gk_y = np.clip(target[1], pitch.goal_y_min - 5, pitch.goal_y_max + 5)
    target = np.array([gk_x, gk_y])

    # --- shot-stopping (predict ball crossing) ---
    if ball.carrier is None and ball.speed > 5:
        toward = ((own_goal_side == 0 and ball.vel[0] < -2)
                  or (own_goal_side == 1 and ball.vel[0] > 2))
        if toward and abs(ball.vel[0]) > 0.1:
            t_line = abs((goal_x - ball.pos[0]) / ball.vel[0])
            if t_line < 3.0:
                pred_y = ball.pos[1] + ball.vel[1] * t_line
                nudge  = 1.0 if own_goal_side == 0 else -1.0
                target = np.array([goal_x + nudge, pred_y])
                return target, True, None

    return target, False, None


# =====================================================================
# Evaluation helpers
# =====================================================================

def _get_pressure(player, opp_team):
    p = 0.0
    for opp in opp_team:
        d = np.linalg.norm(opp.pos - player.pos)
        if d < 10:
            p += (10 - d) / 10
    return min(1.0, p)


def _goal_angle(pos, pitch, atk_dir):
    side = 1 if atk_dir == 1 else 0
    p1 = np.array([0.0 if side == 0 else pitch.length, pitch.goal_y_min])
    p2 = np.array([0.0 if side == 0 else pitch.length, pitch.goal_y_max])
    v1, v2 = p1 - pos, p2 - pos
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(cos_a, -1, 1)) / np.radians(30)


def _shot_blocked(pos, goal, opp_team):
    d = goal - pos
    dl = np.linalg.norm(d)
    if dl < 0.1:
        return False
    d /= dl
    for opp in opp_team:
        t = opp.pos - pos
        proj = np.dot(t, d)
        if 0 < proj < dl and np.linalg.norm(t - proj * d) < 1.5:
            return True
    return False


def _evaluate_pass(passer, target_player, opp_team, pitch, atk_dir, instr):
    dist = np.linalg.norm(target_player.pos - passer.pos)
    if dist < 3 or dist > 55:
        return 0.0

    # directness preference: low directness penalises long passes
    if dist > 25 and instr.directness < 0.3:
        return 0.05

    # passing lane clear?
    pdir = target_player.pos - passer.pos
    pdir_n = pdir / dist
    lane_clear = True
    for opp in opp_team:
        t = opp.pos - passer.pos
        proj = np.dot(t, pdir_n)
        if 0 < proj < dist and np.linalg.norm(t - proj * pdir_n) < 2.0:
            lane_clear = False
            break

    if not lane_clear:
        return 0.05

    # advancement toward goal
    goal = pitch.goal_center(1 if atk_dir == 1 else 0)
    adv  = np.dot(target_player.pos - passer.pos, goal - passer.pos)
    adv /= np.linalg.norm(goal - passer.pos) + 1e-6
    adv_score = np.clip(adv / 30.0 + 0.3, 0, 1)

    # tempo: high tempo rewards forward passes more
    adv_score *= 0.7 + instr.tempo * 0.6

    # receiver in space
    space = min(np.linalg.norm(opp.pos - target_player.pos) for opp in opp_team)
    space_score = np.clip(space / 10.0, 0, 1)

    # chemistry + vision
    chem   = pair_chemistry(passer, target_player)
    vision = passer.attrs.vision / 100.0

    return adv_score * space_score * vision * (0.6 + chem * 0.4)


def _evaluate_dribble(player, opp_team, atk_dir, pitch, instr):
    goal = pitch.goal_center(1 if atk_dir == 1 else 0)
    ahead = goal - player.pos
    an = np.linalg.norm(ahead)
    if an > 0.1:
        ahead /= an

    nearest = 50.0
    for opp in opp_team:
        t = opp.pos - player.pos
        proj = np.dot(t, ahead)
        if 0 < proj < 15 and np.linalg.norm(t - proj * ahead) < 3:
            nearest = min(nearest, proj)

    space = np.clip(nearest / 15.0, 0, 1)
    skill = player.attrs.dribbling / 100.0
    # low directness (build-up) favours dribbling; high directness discourages it
    dir_mod = 1.1 - instr.directness * 0.4
    return space * skill * 0.6 * dir_mod


def _aim_shot(player, pitch, atk_dir, rng):
    side   = 1 if atk_dir == 1 else 0
    goal_x = pitch.length if side == 1 else 0.0
    skill  = player.attrs.accuracy / 100.0
    spread = pitch.goal_width / 2 * (0.5 + 0.4 * skill)
    y      = pitch.width / 2 + rng.choice([-1, 1]) * spread
    return np.array([goal_x, y])


def _lead_pass(passer, receiver, pitch, atk_dir):
    target = receiver.pos.copy()
    if np.linalg.norm(receiver.vel) > 1.0:
        travel = np.linalg.norm(receiver.pos - passer.pos) / 20.0
        target += receiver.vel * travel * 0.5
    return pitch.clamp(target)


def _dribble_target(player, opp_team, pitch, atk_dir):
    goal = pitch.goal_center(1 if atk_dir == 1 else 0)
    d = goal - player.pos
    dn = np.linalg.norm(d)
    if dn > 0.1:
        d /= dn
    target = player.pos + d * 5
    for opp in opp_team:
        if np.linalg.norm(opp.pos - player.pos) < 4:
            perp = np.array([-d[1], d[0]])
            target += perp * 3
            break
    return pitch.clamp(target)

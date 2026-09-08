import numpy as np


class PlayerAttributes:
    """Immutable skill ratings on a 0-100 scale."""

    def __init__(self, **kw):
        # physical
        self.speed        = kw.get("speed", 70)
        self.acceleration = kw.get("acceleration", 70)
        self.stamina      = kw.get("stamina", 70)
        self.strength     = kw.get("strength", 70)
        self.agility      = kw.get("agility", 70)
        # technical
        self.kick_power   = kw.get("kick_power", 70)
        self.accuracy     = kw.get("accuracy", 70)
        self.first_touch  = kw.get("first_touch", 70)
        self.dribbling    = kw.get("dribbling", 70)
        self.tackling     = kw.get("tackling", 70)
        self.heading      = kw.get("heading", 70)
        # mental
        self.vision       = kw.get("vision", 70)
        self.composure    = kw.get("composure", 70)
        self.positioning  = kw.get("positioning", 70)
        self.work_rate    = kw.get("work_rate", 70)
        self.anticipation = kw.get("anticipation", 70)
        # goalkeeper
        self.reflexes     = kw.get("reflexes", 10)
        self.diving       = kw.get("diving", 10)
        self.handling     = kw.get("handling", 10)
        # physical dimensions
        self.height_cm    = kw.get("height_cm", 178)
        self.weight_kg    = kw.get("weight_kg", 75)


class Player:
    """A single player with physics-based movement and kick mechanics."""

    CONTROL_RADIUS = 1.5      # m — can collect a loose ball
    TACKLE_RADIUS  = 2.0      # m — can attempt a tackle

    def __init__(self, name, role, slot, attrs=None):
        self.name  = name
        self.role  = role                        # GK, CB, LB, RB, CDM, CM, …
        self.slot  = slot                        # (x_frac, y_frac) in own half
        self.attrs = attrs or PlayerAttributes()

        # dynamic state
        self.pos             = np.zeros(2)
        self.vel             = np.zeros(2)
        self.stamina_current = float(self.attrs.stamina)
        self.has_ball        = False
        self.team_idx        = 0                 # set by match engine
        self.cooldown        = 0.0               # seconds until next action

        # injury state
        self.injured         = False
        self.injury_severity = 0.0               # 0-1 (0=fine, >0.5=major)

        # decision cache (set each decision tick)
        self._target = None
        self._sprint = False
        self._action = None

    # ── derived physical limits ──────────────────────────────────────

    @property
    def max_sprint_speed(self):
        base = 6.0 + self.attrs.speed * 0.05          # 6 – 11 m/s
        fatigue = max(0.6, self.stamina_current / self.attrs.stamina)
        return base * fatigue

    @property
    def max_jog_speed(self):
        return self.max_sprint_speed * 0.55

    @property
    def accel(self):
        base = 2.0 + self.attrs.acceleration * 0.03   # 2 – 5 m/s²
        weight_factor = 75.0 / max(60, self.attrs.weight_kg)  # heavier = slower accel
        injury_factor = 1.0 - self.injury_severity * 0.5
        return base * min(1.1, weight_factor) * injury_factor

    @property
    def aerial_reach(self):
        """Max height this player can contest a ball (m)."""
        jump_bonus = self.attrs.strength * 0.003 + self.attrs.agility * 0.002
        return self.attrs.height_cm / 100.0 + jump_bonus   # ~1.7 – 2.4 m

    @property
    def max_kick_ball_speed(self):
        return 18.0 + self.attrs.kick_power * 0.17    # 18 – 35 m/s

    @property
    def kick_error_std(self):
        base = np.radians(8.0 - self.attrs.accuracy * 0.07)   # ~8°–1°
        fatigue = max(0.5, self.stamina_current / self.attrs.stamina)
        return base / fatigue

    @property
    def effective_control_radius(self):
        if self.role == "GK":
            return self.CONTROL_RADIUS + self.attrs.diving * 0.02
        return self.CONTROL_RADIUS

    # ── movement ─────────────────────────────────────────────────────

    def move_toward(self, target, dt, sprint=False):
        diff = target - self.pos
        dist = np.linalg.norm(diff)

        if dist < 0.05:
            self.vel *= 0.8
            self._drain_stamina(np.linalg.norm(self.vel), dt)
            return

        direction   = diff / dist
        cap         = self.max_sprint_speed if sprint else self.max_jog_speed
        desired_spd = min(cap, dist / dt) if dt > 0 else cap
        desired_vel = direction * desired_spd

        dv     = desired_vel - self.vel
        dv_mag = np.linalg.norm(dv)
        max_dv = self.accel * dt
        if dv_mag > max_dv:
            dv = dv / dv_mag * max_dv

        self.vel += dv
        spd = np.linalg.norm(self.vel)
        if spd > cap:
            self.vel = self.vel / spd * cap

        self.pos += self.vel * dt
        self._drain_stamina(spd, dt)

    def _drain_stamina(self, speed, dt, temp_factor=1.0):
        # temp_factor > 1 in hot weather, < 1 in cold
        if speed > self.max_jog_speed:
            self.stamina_current -= 0.05 * dt * temp_factor
        elif speed > 1.0:
            self.stamina_current -= 0.005 * dt * temp_factor
        else:
            self.stamina_current += 0.01 * dt
        # injury drains stamina faster
        if self.injured:
            self.stamina_current -= 0.02 * self.injury_severity * dt
        self.stamina_current = np.clip(self.stamina_current, 0.0, float(self.attrs.stamina))

    # ── ball actions ─────────────────────────────────────────────────

    def compute_kick(self, target_2d, power_frac, loft, rng):
        """Return (velocity_3d, spin_3d) for a kick toward *target_2d*."""
        diff = target_2d - self.pos
        dist = np.linalg.norm(diff)
        if dist < 0.01:
            return np.zeros(3), np.zeros(3)

        d = diff / dist

        # accuracy noise
        err  = rng.normal(0, self.kick_error_std)
        c, s = np.cos(err), np.sin(err)
        d    = np.array([d[0]*c - d[1]*s, d[0]*s + d[1]*c])

        ball_spd = self.max_kick_ball_speed * np.clip(power_frac, 0.1, 1.0)
        elev     = loft * np.radians(40)                    # 0 → ground, 1 → 40°
        h_spd    = ball_spd * np.cos(elev)
        v_spd    = ball_spd * np.sin(elev)

        return np.array([d[0]*h_spd, d[1]*h_spd, v_spd]), np.zeros(3)

    def attempt_tackle(self, carrier, rng):
        # strength & weight advantage in physical challenges
        phys_adv = (self.attrs.strength - carrier.attrs.strength) * 0.15 \
                 + (self.attrs.weight_kg - carrier.attrs.weight_kg) * 0.1
        t_score = self.attrs.tackling + self.attrs.anticipation * 0.2 + phys_adv + rng.normal(0, 10)
        d_score = carrier.attrs.dribbling + carrier.attrs.agility * 0.1 + rng.normal(0, 10)
        return t_score > d_score

    def check_injury(self, intensity, rng):
        """Roll for injury after a physical action. intensity 0-1."""
        # base probability very low; rises with fatigue, intensity, prior injury
        fatigue = 1.0 - self.stamina_current / max(1, self.attrs.stamina)
        prob = 0.0005 * intensity * (1.0 + fatigue * 2.0)
        if self.injured:
            prob *= 3.0   # re-injury risk
        if rng.random() < prob:
            self.injured = True
            self.injury_severity = min(1.0, self.injury_severity + rng.uniform(0.2, 0.6))
            return True
        return False

    # ── positioning ──────────────────────────────────────────────────

    def formation_pos(self, pitch_length, pitch_width, atk_dir, ball_x, ball_y):
        fx, fy = self.slot
        bxf = ball_x / pitch_length
        byf = ball_y / pitch_width

        shift_x = (bxf - 0.5) * 0.30
        shift_y = (byf - 0.5) * 0.15

        if atk_dir == 1:
            x = (fx + shift_x) * pitch_length
        else:
            x = (1.0 - fx + shift_x) * pitch_length

        y = (fy + shift_y) * pitch_width

        x = np.clip(x, 2.0, pitch_length - 2.0)
        y = np.clip(y, 2.0, pitch_width  - 2.0)
        return np.array([x, y])

"""
Match engine  - the game loop that wires physics, players, and tactics.

All outcomes emerge from the simulation: a shot scores because its
trajectory enters the goal before the GK can physically reach it,
not because a dice roll said so.
"""
import numpy as np

from .ball import Ball
from .pitch import Pitch, Environment
from .player import Player
from . import tactics as ai
from .tactics import ManagerInstructions


# =====================================================================
# Data structures
# =====================================================================

class Team:
    def __init__(self, name, players, instructions=None):
        self.name = name
        self.players = players
        self.instructions = instructions or ManagerInstructions()
        self.score = 0
        self.attack_goal_side = 1          # 0=left, 1=right (swapped at HT)


class MatchEvent:
    def __init__(self, time, etype, **data):
        self.time = time
        self.type = etype
        self.data = data

    def __repr__(self):
        m, s = int(self.time // 60), int(self.time % 60)
        if self.type == "goal":
            return (f"  [{m}'] GOAL! {self.data['scorer']} "
                    f"({self.data['team']})  {self.data['score']}")
        if self.type == "shot":
            return (f"  [{m}'] Shot by {self.data['player']} "
                    f"({self.data['team']})  - {self.data['result']}")
        if self.type == "injury":
            return (f"  [{m}'] INJURY: {self.data['player']} "
                    f"({self.data['team']}) severity {self.data['severity']:.0%}")
        if self.type == "info":
            return f"  [{m}'] {self.data.get('msg', '')}"
        return f"  [{m}'] {self.type} {self.data}"


# =====================================================================
# Match
# =====================================================================

class Match:
    PHYSICS_DT          = 1.0 / 30.0    # 30 Hz  (fast enough for ball control)
    DECISION_INTERVAL   = 6             # ticks between AI decisions (~5 Hz)
    HALF_DURATION       = 45 * 60       # seconds

    def __init__(self, team_a, team_b, pitch=None, env=None, seed=42):
        self.team_a = team_a
        self.team_b = team_b
        self.pitch  = pitch or Pitch()
        self.env    = env or Environment()
        self.ball   = Ball()
        self.rng    = np.random.RandomState(seed)
        self.time   = 0.0
        self.half   = 1
        self.events = []

        # stats
        self.possession_ticks = [0, 0]
        self.shots            = [0, 0]
        self.shots_on_target  = [0, 0]
        self.passes_attempted = [0, 0]
        self.tackles           = [0, 0]
        self.injuries          = [0, 0]
        self.last_possession   = 0

        # temperature → stamina multiplier (hotter = harder)
        self.temp_factor = 1.0 + max(0, self.env.temperature - 22) * 0.03

        # assign team indices
        for p in team_a.players:
            p.team_idx = 0
        for p in team_b.players:
            p.team_idx = 1

        team_a.attack_goal_side = 1
        team_b.attack_goal_side = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _all_players(self):
        return self.team_a.players + self.team_b.players

    def _team_of(self, player):
        return self.team_a if player.team_idx == 0 else self.team_b

    def _opp_of(self, player):
        return self.team_b.players if player.team_idx == 0 else self.team_a.players

    def _own_of(self, player):
        return self.team_a.players if player.team_idx == 0 else self.team_b.players

    def _atk_dir(self, player):
        t = self._team_of(player)
        return 1 if t.attack_goal_side == 1 else -1

    def _instr(self, player):
        return self._team_of(player).instructions

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _setup_kickoff(self, kicking_team_idx):
        self.ball.place(self.pitch.length / 2, self.pitch.width / 2)
        self.ball.carrier = None

        for p in self._all_players():
            d = self._atk_dir(p)
            p.pos = p.formation_pos(self.pitch.length, self.pitch.width, d,
                                    self.pitch.length / 2, self.pitch.width / 2)
            p.vel[:] = 0
            p.has_ball = False
            p.cooldown = 0.0

        # give ball to a forward on the kicking team
        kt = self.team_a if kicking_team_idx == 0 else self.team_b
        for p in kt.players:
            if p.role in ("ST", "CF", "CAM", "RW", "LW"):
                p.pos = np.array([self.pitch.length / 2, self.pitch.width / 2])
                self.ball.carrier = p
                p.has_ball = True
                break

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self):
        """Simulate a full match.  Returns list of MatchEvents."""
        self.half = 1
        self._setup_kickoff(0)
        self.events.append(MatchEvent(0, "info", msg="Kick-off  - first half"))
        self._run_half()
        self.events.append(MatchEvent(self.time, "info", msg="Half-time"))

        # switch sides
        self.team_a.attack_goal_side = 0
        self.team_b.attack_goal_side = 1
        self.half = 2
        self.time = self.HALF_DURATION
        self._setup_kickoff(1)
        self.events.append(MatchEvent(self.time, "info", msg="Kick-off  - second half"))
        self._run_half()
        self.events.append(MatchEvent(
            self.time, "info",
            msg=f"Full-time: {self.team_a.name} {self.team_a.score} - "
                f"{self.team_b.score} {self.team_b.name}"))

        return self.events

    def _run_half(self):
        end = self.time + self.HALF_DURATION
        tick = 0
        while self.time < end:
            if tick % self.DECISION_INTERVAL == 0:
                self._decision_phase()
            self._action_phase()
            self._ball_physics()
            self._possession_phase()
            self._rules_phase()
            self._sync()
            self.time += self.PHYSICS_DT
            tick += 1

    # ------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------

    def _decision_phase(self):
        for p in self._all_players():
            target, sprint, action = ai.get_decision(
                p, self.ball, self._own_of(p), self._opp_of(p),
                self.pitch, self._atk_dir(p), self._instr(p), self.rng)
            p._target = target
            p._sprint = sprint
            p._action = action

    def _action_phase(self):
        dt = self.PHYSICS_DT
        for p in self._all_players():
            # movement
            if p._target is not None:
                p.move_toward(p._target, dt, p._sprint)
                p.pos = self.pitch.clamp(p.pos)

            # cooldowns
            if p.cooldown > 0:
                p.cooldown -= dt
                continue

            # ball action
            act = p._action
            if act is None:
                continue

            if act[0] == "shoot" and p.has_ball:
                _, target, power, loft = act
                vel, spin = p.compute_kick(target, power, loft, self.rng)
                self.ball.kick(vel, spin)
                p.has_ball = False
                p.cooldown = 0.5
                p._action  = None

                team = self._team_of(p)
                self.shots[p.team_idx] += 1

                # check if on-target (predict crossing)
                goal_side = team.attack_goal_side
                goal_x = self.pitch.length if goal_side == 1 else 0.0
                if abs(vel[0]) > 0.1:
                    t_cross = (goal_x - self.ball.pos[0]) / vel[0]
                    if t_cross > 0:
                        py = self.ball.pos[1] + vel[1] * t_cross
                        pz = max(0, self.ball.pos[2] + vel[2] * t_cross
                                 - 0.5 * 9.81 * t_cross ** 2)
                        on_frame = (self.pitch.goal_y_min <= py <= self.pitch.goal_y_max
                                    and pz <= self.pitch.goal_height)
                        if on_frame:
                            self.shots_on_target[p.team_idx] += 1
                            result = "on target"
                        else:
                            result = "off target"
                    else:
                        result = "off target"
                else:
                    result = "off target"

                self.events.append(MatchEvent(self.time, "shot",
                    player=p.name, team=team.name, result=result))

                # injury risk on powerful shot
                p.check_injury(power * 0.3, self.rng)

            elif act[0] == "pass" and p.has_ball:
                _, target, power, loft = act
                vel, spin = p.compute_kick(target, power, loft, self.rng)
                self.ball.kick(vel, spin)
                p.has_ball = False
                p.cooldown = 0.3
                p._action  = None
                self.passes_attempted[p.team_idx] += 1

            elif act[0] == "tackle" and not p.has_ball:
                if (self.ball.carrier is not None
                        and self.ball.carrier.team_idx != p.team_idx):
                    carrier = self.ball.carrier
                    dist = np.linalg.norm(p.pos - carrier.pos)
                    if dist < p.TACKLE_RADIUS:
                        self.tackles[p.team_idx] += 1
                        if p.attempt_tackle(carrier, self.rng):
                            carrier.has_ball = False
                            self.ball.carrier = None
                            self.ball.vel[:2] = self.rng.normal(0, 2, 2)
                            self.ball.vel[2]  = 0
                        p.cooldown      = 0.8
                        carrier.cooldown = 0.5
                        # injury risk from challenge
                        intensity = 0.4 + p.attrs.strength * 0.004
                        p.check_injury(intensity * 0.5, self.rng)
                        got_hurt = carrier.check_injury(intensity, self.rng)
                        if got_hurt:
                            t = self._team_of(carrier)
                            self.injuries[carrier.team_idx] += 1
                            self.events.append(MatchEvent(self.time, "injury",
                                player=carrier.name, team=t.name,
                                severity=carrier.injury_severity))
                p._action = None

    def _ball_physics(self):
        if self.ball.carrier is not None:
            c = self.ball.carrier
            self.ball.pos[0] = c.pos[0]
            self.ball.pos[1] = c.pos[1]
            self.ball.pos[2] = self.ball.RADIUS
            self.ball.vel[:2] = c.vel
            self.ball.vel[2]  = 0.0
        self.ball.update(self.PHYSICS_DT, self.env)

    def _possession_phase(self):
        # track possession
        if self.ball.carrier is not None:
            self.possession_ticks[self.ball.carrier.team_idx] += 1
            self.last_possession = self.ball.carrier.team_idx
            return

        ball_xy = self.ball.xy
        ball_z  = self.ball.pos[2]
        best, best_d = None, float("inf")

        for p in self._all_players():
            if p.cooldown > 0:
                continue
            # height check  - can this player reach the ball?
            if ball_z > p.aerial_reach:
                continue
            d = np.linalg.norm(p.pos - ball_xy)
            r = p.effective_control_radius
            if d < r and d < best_d:
                best_d = d
                best   = p

        if best is not None:
            self.ball.carrier = best
            best.has_ball     = True
            self.ball.vel[:]  = 0
            self.ball.spin[:] = 0

    def _rules_phase(self):
        # --- goal check ---
        gs = self.pitch.is_in_goal(self.ball.pos)
        if gs >= 0:
            if gs == 0:
                scoring = (self.team_a
                           if self.team_a.attack_goal_side == 0
                           else self.team_b)
            else:
                scoring = (self.team_a
                           if self.team_a.attack_goal_side == 1
                           else self.team_b)
            scoring.score += 1

            scorer = "Unknown"
            for e in reversed(self.events):
                if e.type == "shot":
                    scorer = e.data["player"]
                    break

            self.events.append(MatchEvent(self.time, "goal",
                scorer=scorer, team=scoring.name,
                score=f"{self.team_a.score}-{self.team_b.score}"))

            non_scoring = 0 if scoring is self.team_b else 1
            self._setup_kickoff(non_scoring)
            return

        # --- out of bounds ---
        oob = self.pitch.out_of_bounds(self.ball.pos[:2])
        if oob and self.ball.carrier is None:
            # restart: give to team that didn't touch it last
            restart_team = 1 - self.last_possession
            rx = np.clip(self.ball.pos[0], 2.0, self.pitch.length - 2.0)
            ry = np.clip(self.ball.pos[1], 2.0, self.pitch.width  - 2.0)
            self.ball.place(rx, ry)

            rt = self.team_a if restart_team == 0 else self.team_b
            best, best_d = None, float("inf")
            for p in rt.players:
                d = np.linalg.norm(p.pos - self.ball.xy)
                if d < best_d:
                    best_d = d
                    best   = p
            if best:
                self.ball.carrier = best
                best.has_ball     = True

    def _sync(self):
        for p in self._all_players():
            p.has_ball = (self.ball.carrier is p)

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self):
        total = max(1, self.possession_ticks[0] + self.possession_ticks[1])
        poss  = [self.possession_ticks[i] / total * 100 for i in range(2)]
        lines = [
            "",
            f"  {'Stat':<22} {self.team_a.name:>14}   {self.team_b.name:>14}",
            f"  {'-'*52}",
            f"  {'Possession':<22} {poss[0]:>13.0f}%   {poss[1]:>13.0f}%",
            f"  {'Shots':<22} {self.shots[0]:>14}   {self.shots[1]:>14}",
            f"  {'Shots on target':<22} {self.shots_on_target[0]:>14}   {self.shots_on_target[1]:>14}",
            f"  {'Passes attempted':<22} {self.passes_attempted[0]:>14}   {self.passes_attempted[1]:>14}",
            f"  {'Tackles':<22} {self.tackles[0]:>14}   {self.tackles[1]:>14}",
            f"  {'Injuries':<22} {self.injuries[0]:>14}   {self.injuries[1]:>14}",
        ]
        return "\n".join(lines)

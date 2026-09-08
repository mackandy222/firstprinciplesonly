import numpy as np


class Pitch:
    """Pitch geometry and boundary checks."""

    def __init__(self, length=105.0, width=68.0):
        self.length = length
        self.width = width
        self.goal_width = 7.32
        self.goal_height = 2.44

        # derived
        self.center = np.array([length / 2.0, width / 2.0])
        self.goal_y_min = (width - self.goal_width) / 2.0
        self.goal_y_max = (width + self.goal_width) / 2.0

    def goal_center(self, side):
        """side 0 = left goal (x=0), side 1 = right goal (x=length)."""
        return np.array([0.0 if side == 0 else self.length, self.width / 2.0])

    def is_in_goal(self, pos):
        """Returns goal side (0 or 1) if ball is inside a goal, else -1."""
        z = pos[2] if len(pos) > 2 else 0.0
        if not (self.goal_y_min <= pos[1] <= self.goal_y_max and z <= self.goal_height):
            return -1
        if pos[0] <= 0.0:
            return 0
        if pos[0] >= self.length:
            return 1
        return -1

    def out_of_bounds(self, xy):
        if xy[0] < 0:
            return "goal_line_left"
        if xy[0] > self.length:
            return "goal_line_right"
        if xy[1] < 0 or xy[1] > self.width:
            return "sideline"
        return None

    def clamp(self, xy):
        return np.array([
            np.clip(xy[0], 1.0, self.length - 1.0),
            np.clip(xy[1], 1.0, self.width - 1.0),
        ])


class Environment:
    """Weather and surface constants fed into ball physics each tick."""

    def __init__(self):
        self.air_density = 1.225       # kg/m³ (sea level, 15 °C)
        self.wind = np.zeros(3)        # m/s
        self.ground_friction = 0.4     # rolling-friction coefficient
        self.restitution = 0.55        # coefficient of restitution (bounce)
        self.drag_coeff = 0.25         # ball Cd
        self.magnus_coeff = 0.25       # ball Cl (spin)
        self.temperature = 20.0        # °C

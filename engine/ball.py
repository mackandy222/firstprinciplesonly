import numpy as np


class Ball:
    """Football with full 3D physics: gravity, drag, Magnus spin, bounce, roll."""

    MASS = 0.43                                  # kg  (FIFA: 410-450 g)
    RADIUS = 0.11                                # m   (FIFA: 68-70 cm circumference)
    CROSS_AREA = np.pi * RADIUS ** 2             # frontal area

    def __init__(self):
        self.pos = np.zeros(3)                   # (x, y, z) metres
        self.vel = np.zeros(3)                   # m/s
        self.spin = np.zeros(3)                  # rad/s  (angular velocity)
        self.carrier = None                      # Player holding the ball

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def place(self, x, y, z=None):
        self.pos[:] = [x, y, z if z is not None else self.RADIUS]
        self.vel[:] = 0
        self.spin[:] = 0

    @property
    def xy(self):
        return self.pos[:2].copy()

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))

    @property
    def is_airborne(self):
        return self.pos[2] > self.RADIUS + 0.3

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def kick(self, velocity, spin=None):
        """Set ball velocity and spin (called by a player kick)."""
        self.vel = np.asarray(velocity, dtype=float).copy()
        self.spin = np.asarray(spin, dtype=float).copy() if spin is not None else np.zeros(3)
        self.carrier = None

    # ------------------------------------------------------------------
    # physics step
    # ------------------------------------------------------------------

    def update(self, dt, env):
        if self.carrier is not None:
            return                               # ball moves with carrier

        force = np.zeros(3)

        # -- gravity --
        force[2] -= self.MASS * 9.81

        # -- aerodynamic forces --
        v_rel = self.vel - env.wind
        spd = np.linalg.norm(v_rel)

        if spd > 0.01:
            v_hat = v_rel / spd
            # drag:  F = -½ ρ Cd A |v|² v̂
            force -= 0.5 * env.air_density * env.drag_coeff * self.CROSS_AREA * spd**2 * v_hat
            # Magnus: F = Cm (ω × v) ρ A r
            force += env.magnus_coeff * np.cross(self.spin, v_rel) \
                     * env.air_density * self.CROSS_AREA * self.RADIUS

        # -- integrate (semi-implicit Euler) --
        self.vel += (force / self.MASS) * dt
        self.pos += self.vel * dt

        # -- spin decay --
        if np.linalg.norm(self.spin) > 0.1:
            self.spin *= max(0.0, 1.0 - 2.0 * dt)

        # -- ground interaction --
        if self.pos[2] < self.RADIUS:
            self.pos[2] = self.RADIUS

            if self.vel[2] < -0.3:
                # bounce
                self.vel[2] *= -env.restitution
                self.vel[0] *= 1.0 - env.ground_friction * 0.3
                self.vel[1] *= 1.0 - env.ground_friction * 0.3
            else:
                # rolling
                self.vel[2] = 0.0
                gspd = np.linalg.norm(self.vel[:2])
                if gspd > 0.05:
                    f = max(0.0, 1.0 - env.ground_friction * 9.81 * dt / gspd)
                    self.vel[0] *= f
                    self.vel[1] *= f
                else:
                    self.vel[:2] = 0.0

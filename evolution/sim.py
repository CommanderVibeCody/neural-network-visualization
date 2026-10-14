"""Vectorized 2D car simulation: bicycle-style heading model, ray sensors, gates, deaths."""
import numpy as np

V_MAX = 170.0        # px / s
ACCEL = 260.0        # px / s^2
MAX_STEER = 0.55     # rad
WHEEL_BASE = 22.0
RADIUS = 6.0
STUCK_TIME = 3.5     # s without passing a gate -> dead


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def segments_intersect(p1, p2, q1, q2):
    """Vectorized proper intersection test for N segment pairs (each (N,2))."""
    r = p2 - p1
    s = q2 - q1
    den = _cross(r[:, 0], r[:, 1], s[:, 0], s[:, 1])
    qp = q1 - p1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(qp[:, 0], qp[:, 1], s[:, 0], s[:, 1]) / den
        u = _cross(qp[:, 0], qp[:, 1], r[:, 0], r[:, 1]) / den
    return (np.abs(den) > 1e-9) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)


class Sim:
    def __init__(self, track, n, n_rays=10, ray_len=220.0, speed_input=True, throttle_mode="tanh"):
        self.track, self.n, self.n_rays = track, n, n_rays
        self.ray_len, self.speed_input = ray_len, speed_input
        self.throttle_mode = throttle_mode  # "tanh": [-1,1] (reverse possible), "sigmoid": [0,1]
        self.ray_angles = np.deg2rad(np.linspace(-90, 90, n_rays))
        w = track.walls
        self.A, self.E = w[:, :2], w[:, 2:] - w[:, :2]
        self.reset()

    @property
    def n_inputs(self):
        return self.n_rays + (1 if self.speed_input else 0)

    def reset(self):
        n, tr = self.n, self.track
        self.pos = np.tile(tr.start_pos, (n, 1)).astype(float)
        self.heading = np.full(n, tr.start_heading)
        self.v = np.zeros(n)
        self.alive = np.ones(n, bool)
        self.passed = np.zeros(n, int)
        self.last_gate_t = np.zeros(n)
        self.speed_sum = np.zeros(n)
        self.steps_alive = np.zeros(n)
        self.fitness = np.zeros(n)
        self.t = 0.0
        self.out = np.zeros((n, 2))
        self.sense()

    def sense(self):
        """Cast rays; returns normalized distances (N, R) in [0,1] and stores hit points."""
        ang = self.heading[:, None] + self.ray_angles[None, :]
        D = np.stack([np.cos(ang), np.sin(ang)], axis=-1)            # (N,R,2)
        AP = self.A[None, :, :] - self.pos[:, None, :]                # (N,M,2)
        E = self.E
        den = D[..., 0, None] * E[None, None, :, 1] - D[..., 1, None] * E[None, None, :, 0]  # (N,R,M)
        num_t = (AP[:, :, 0] * E[None, :, 1] - AP[:, :, 1] * E[None, :, 0])[:, None, :]     # (N,1,M)
        num_u = AP[:, None, :, 0] * D[..., 1, None] - AP[:, None, :, 1] * D[..., 0, None]   # (N,R,M)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num_t / den
            u = num_u / den
        valid = (np.abs(den) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(valid, t, np.inf).min(axis=2)
        t = np.minimum(t, self.ray_len)
        self.ray_dist = t / self.ray_len
        self.ray_hits = self.pos[:, None, :] + D * t[..., None]
        return self.ray_dist

    def inputs(self):
        x = self.ray_dist
        if self.speed_input:
            x = np.hstack([x, (self.v / V_MAX)[:, None]])
        return x

    def _wall_distance(self):
        AP = self.pos[:, None, :] - self.A[None, :, :]               # (N,M,2)
        EE = (self.E ** 2).sum(1) + 1e-9
        u = np.clip((AP * self.E[None]).sum(2) / EE, 0, 1)
        closest = self.A[None] + u[..., None] * self.E[None]
        return np.linalg.norm(self.pos[:, None, :] - closest, axis=2).min(axis=1)

    def step(self, actions, dt=1 / 30):
        """actions: (N,2) = [throttle, steering]; only alive cars move."""
        a = self.alive
        if not a.any():
            return
        self.out = actions
        thr, steer = actions[:, 0], np.clip(actions[:, 1], -1, 1)
        target = np.clip(thr, -1, 1) * V_MAX
        if self.throttle_mode == "tanh":
            target = np.where(target < 0, target * 0.4, target)  # weak reverse
        dv = np.clip(target - self.v, -ACCEL * dt, ACCEL * dt)
        self.v = np.where(a, self.v + dv, self.v)
        self.heading = np.where(a, self.heading + self.v / WHEEL_BASE * np.tan(steer * MAX_STEER) * dt,
                                self.heading)
        old = self.pos.copy()
        step = np.stack([np.cos(self.heading), np.sin(self.heading)], 1) * (self.v * dt)[:, None]
        self.pos = np.where(a[:, None], self.pos + step, self.pos)
        self.t += dt

        # Gates: forward crossing of next gate / backward crossing of the last passed gate
        G = self.track.n_gates
        nxt = self.track.gates[(self.passed + 1) % G]
        prv = self.track.gates[self.passed % G]
        fwd = segments_intersect(old, self.pos, nxt[:, 0], nxt[:, 1]) & a
        back = segments_intersect(old, self.pos, prv[:, 0], prv[:, 1]) & a & (self.v < 0)
        self.passed += fwd
        self.last_gate_t = np.where(fwd, self.t, self.last_gate_t)

        crash = self._wall_distance() < RADIUS
        stuck = (self.t - self.last_gate_t) > STUCK_TIME
        self.alive = a & ~crash & ~stuck & ~back

        self.speed_sum += np.where(a, np.maximum(self.v, 0), 0)
        self.steps_alive += a
        self._update_fitness()
        self.sense()

    def _update_fitness(self):
        """Fitness = gates passed (laps included) + fraction to next gate + small speed bonus."""
        G = self.track.n_gates
        m_prev = self.track.gate_mid[self.passed % G]
        m_next = self.track.gate_mid[(self.passed + 1) % G]
        seg = np.linalg.norm(m_next - m_prev, axis=1) + 1e-9
        frac = np.clip(1 - np.linalg.norm(m_next - self.pos, axis=1) / seg, 0, 1)
        avg_speed = self.speed_sum / np.maximum(self.steps_alive, 1) / V_MAX
        self.fitness = self.passed + frac + 0.5 * avg_speed

    @property
    def laps(self):
        return self.passed // self.track.n_gates

"""N-body gravitational system: forces, energy, momentum, and time stepping.

Units are chosen with G = 1 (standard in celestial-mechanics test problems).
Softening avoids the r -> 0 singularity for close encounters.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from integrators import INTEGRATORS

Vec = List[float]


class NBody:
    def __init__(self, masses: List[float], pos: List[Vec], vel: List[Vec],
                 G: float = 1.0, softening: float = 0.0):
        assert len(masses) == len(pos) == len(vel)
        self.m = list(masses)
        self.pos = [list(p) for p in pos]
        self.vel = [list(v) for v in vel]
        self.G = G
        self.soft2 = softening * softening
        self.n = len(masses)

    def accel(self, pos: List[Vec]) -> List[Vec]:
        """Newtonian gravitational acceleration on each body: a_i = G * sum_j m_j (r_j - r_i)/|r|^3."""
        n = self.n
        a = [[0.0, 0.0, 0.0] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                dz = pos[j][2] - pos[i][2]
                r2 = dx * dx + dy * dy + dz * dz + self.soft2
                inv_r3 = 1.0 / (r2 * math.sqrt(r2))
                # Newton's third law: equal and opposite, mass-weighted.
                fi = self.G * self.m[j] * inv_r3
                fj = self.G * self.m[i] * inv_r3
                a[i][0] += fi * dx; a[i][1] += fi * dy; a[i][2] += fi * dz
                a[j][0] -= fj * dx; a[j][1] -= fj * dy; a[j][2] -= fj * dz
        return a

    def kinetic_energy(self) -> float:
        return 0.5 * sum(
            self.m[i] * (self.vel[i][0] ** 2 + self.vel[i][1] ** 2 + self.vel[i][2] ** 2)
            for i in range(self.n)
        )

    def potential_energy(self) -> float:
        u = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dx = self.pos[j][0] - self.pos[i][0]
                dy = self.pos[j][1] - self.pos[i][1]
                dz = self.pos[j][2] - self.pos[i][2]
                r = math.sqrt(dx * dx + dy * dy + dz * dz + self.soft2)
                u -= self.G * self.m[i] * self.m[j] / r
        return u

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def linear_momentum(self) -> Vec:
        p = [0.0, 0.0, 0.0]
        for i in range(self.n):
            for k in range(3):
                p[k] += self.m[i] * self.vel[i][k]
        return p

    def angular_momentum(self) -> Vec:
        L = [0.0, 0.0, 0.0]
        for i in range(self.n):
            r, v, m = self.pos[i], self.vel[i], self.m[i]
            L[0] += m * (r[1] * v[2] - r[2] * v[1])
            L[1] += m * (r[2] * v[0] - r[0] * v[2])
            L[2] += m * (r[0] * v[1] - r[1] * v[0])
        return L

    def step(self, method: str, dt: float) -> None:
        integrator = INTEGRATORS[method]
        self.pos, self.vel = integrator(self.pos, self.vel, self.accel, dt)

    def run(self, method: str, dt: float, steps: int, sample_every: int = 1):
        """Integrate and yield (t, energy, |L|) diagnostics every `sample_every` steps."""
        t = 0.0
        L0 = self.angular_momentum()
        for s in range(steps):
            self.step(method, dt)
            t += dt
            if s % sample_every == 0:
                yield t, self.total_energy(), _norm(self.angular_momentum())


def _norm(v: Vec) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

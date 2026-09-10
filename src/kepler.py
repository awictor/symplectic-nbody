"""Exact analytic two-body (Kepler) solution -- the ground truth.

For two bodies the orbit is a conic section with a closed-form time dependence,
obtained by solving Kepler's equation M = E - e sin E for the eccentric anomaly
E. This gives an *exact* position at any time t, with no accumulated numerical
error, so it is the perfect yardstick for measuring an integrator's real error
and confirming its convergence order.

Everything is dependency-free stdlib math.
"""

from __future__ import annotations

import math
from typing import Tuple


def solve_kepler(M: float, e: float, tol: float = 1e-14, max_iter: int = 100) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for the eccentric anomaly E
    via Newton-Raphson. Converges quadratically for e < 1."""
    M = math.fmod(M, 2.0 * math.pi)
    E = M if e < 0.8 else math.pi  # good initial guess
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            break
    return E


class KeplerOrbit:
    """Analytic relative orbit for a two-body system, in the orbital plane.

    Parameterized the same way as systems.two_body_eccentric: bodies start at
    apoapsis on the x-axis with the relative orbit in the xy-plane.
    """

    def __init__(self, e: float = 0.7, a: float = 1.0, mu: float = 2.0):
        assert 0.0 <= e < 1.0
        self.e = e
        self.a = a
        self.mu = mu
        self.n = math.sqrt(mu / a ** 3)  # mean motion
        self.period = 2.0 * math.pi / self.n
        # Start at apoapsis: true anomaly = pi, so E starts at pi (M=pi).
        self.M0 = math.pi

    def relative_state(self, t: float) -> Tuple[float, float, float, float]:
        """Return (x, y, vx, vy) of body-2 relative to body-1 at time t."""
        M = self.M0 + self.n * t
        E = solve_kepler(M, self.e)
        a, e, mu = self.a, self.e, self.mu
        cosE, sinE = math.cos(E), math.sin(E)
        r = a * (1.0 - e * cosE)
        # position in perifocal frame (periapsis along +x). We started at
        # apoapsis on +x, so rotate the perifocal frame by pi.
        px = a * (cosE - e)
        py = a * math.sqrt(1.0 - e * e) * sinE
        # velocity in perifocal frame
        Edot = self.n / (1.0 - e * cosE)
        vpx = -a * sinE * Edot
        vpy = a * math.sqrt(1.0 - e * e) * cosE * Edot
        # rotate by pi (apoapsis-start convention): (x,y) -> (-x,-y)
        return -px, -py, -vpx, -vpy

    def barycentric(self, t: float, m1: float, m2: float):
        """Return per-body (pos, vel) lists matching the NBody layout, with the
        barycentre fixed at the origin (as two_body_eccentric sets it up)."""
        x, y, vx, vy = self.relative_state(t)
        mu_tot = m1 + m2
        # r1 = -m2/M * r_rel, r2 = +m1/M * r_rel
        p1 = [-m2 / mu_tot * x, -m2 / mu_tot * y, 0.0]
        p2 = [m1 / mu_tot * x, m1 / mu_tot * y, 0.0]
        v1 = [-m2 / mu_tot * vx, -m2 / mu_tot * vy, 0.0]
        v2 = [m1 / mu_tot * vx, m1 / mu_tot * vy, 0.0]
        return [p1, p2], [v1, v2]

"""The Sitnikov problem: the cleanest route to chaos in celestial mechanics.

Two equal masses orbit their common barycentre on a Keplerian ellipse in the
xy-plane. A third, massless body is constrained to the z-axis through the
barycentre by symmetry. Its entire dynamics is one non-autonomous 1-D equation:

    z'' = - z / (z^2 + r(t)^2)^{3/2}

where r(t) is the (time-varying) distance of each primary from the barycentre.
With G = 1 and total primary mass = 1, each primary has mass 1/2 and sits at
distance rho(t)/2, where rho(t) is the binary separation on an orbit of
semi-major axis a = 1 and eccentricity e; so r(t) = rho(t)/2.

  * e = 0 (circular binary): the forcing is constant, the system is integrable,
    every bounded orbit is periodic/quasi-periodic.
  * e > 0: the periodic forcing makes it a textbook chaotic system -- Moser used
    it to prove the existence of chaotic (symbolic-dynamics) orbits.

Same z-equation, one parameter e, a whole bifurcation from order to chaos. The
binary separation comes from the exact Kepler solution (reused from kepler.py).
Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from kepler import solve_kepler


def binary_separation(t: float, e: float, a: float = 1.0, mu: float = 1.0) -> float:
    """Separation rho(t) of the two primaries (total mass mu) on a Kepler orbit
    of semi-major axis a, eccentricity e, with period 2*pi*sqrt(a^3/mu)."""
    n = math.sqrt(mu / a ** 3)          # mean motion
    M = n * t                            # mean anomaly (periapsis at t=0)
    E = solve_kepler(M, e)
    return a * (1.0 - e * math.cos(E))   # instantaneous separation


def accel(z: float, t: float, e: float, a: float = 1.0, mu: float = 1.0) -> float:
    """z'' for the test particle. Each primary is at distance r = rho/2 from the
    barycentre, so the on-axis pull is -z / (z^2 + r^2)^{3/2} summed over both
    (mass 1/2 each) -> -z / (z^2 + r^2)^{3/2} with r = rho/2."""
    r = 0.5 * binary_separation(t, e, a, mu)
    denom = (z * z + r * r) ** 1.5
    return -z / denom


def step_rk4(z: float, vz: float, t: float, dt: float, e: float) -> Tuple[float, float]:
    def f(z, vz, t):
        return vz, accel(z, t, e)

    k1z, k1v = f(z, vz, t)
    k2z, k2v = f(z + 0.5 * dt * k1z, vz + 0.5 * dt * k1v, t + 0.5 * dt)
    k3z, k3v = f(z + 0.5 * dt * k2z, vz + 0.5 * dt * k2v, t + 0.5 * dt)
    k4z, k4v = f(z + dt * k3z, vz + dt * k3v, t + dt)
    z2 = z + dt / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z)
    v2 = vz + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return z2, v2


def trajectory(z0: float, vz0: float, e: float, dt: float, n_steps: int):
    """Return (times, z, vz) for the Sitnikov test particle."""
    z, vz, t = z0, vz0, 0.0
    ts, zs, vs = [0.0], [z0], [vz0]
    for _ in range(n_steps):
        z, vz = step_rk4(z, vz, t, dt, e)
        t += dt
        ts.append(t); zs.append(z); vs.append(vz)
    return ts, zs, vs


def poincare_map(z0: float, vz0: float, e: float, n_periods: int = 300,
                 steps_per_period: int = 2000, a: float = 1.0, mu: float = 1.0):
    """Stroboscopic map: sample (z, vz) once per binary period. For e>0 this is
    the natural Poincare section of the periodically-forced system."""
    period = 2.0 * math.pi * math.sqrt(a ** 3 / mu)
    dt = period / steps_per_period
    z, vz, t = z0, vz0, 0.0
    pts: List[Tuple[float, float]] = []
    for _p in range(n_periods):
        for _ in range(steps_per_period):
            z, vz = step_rk4(z, vz, t, dt, e)
            t += dt
            if abs(z) > 1e4:        # escaped
                return pts
        pts.append((z, vz))
    return pts

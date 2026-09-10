"""Mean-motion resonances: when orbital periods lock into integer ratios.

Two planets are in a p:q mean-motion resonance (MMR) when their periods satisfy
T_outer / T_inner ~ p/q (p>q). At exact resonance a "resonant argument" like

    phi = p * lambda_outer - q * lambda_inner - (p - q) * varpi_inner

stops circulating through 2*pi and instead *librates* about a fixed value -- the
dynamical signature of resonance lock. Resonances sculpt the solar system: the
Kirkwood gaps in the asteroid belt (3:1, 5:2 with Jupiter), the Laplace 4:2:1
chain of Io-Europa-Ganymede, Neptune-Pluto 3:2.

This module integrates a real Sun + two-planet system (reusing NBody) and
extracts orbital elements and the resonant argument each step. A librating phi
(bounded range) means the pair is locked; a circulating phi (covers 0..2*pi)
means it is not.

Pure stdlib; the Sun-relative elements come from the standard vis-viva / angular
momentum formulas.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from nbody import NBody

G_DEFAULT = 4.0 * math.pi ** 2  # AU, yr, solar masses


def _elements(pos, vel, mu):
    """Return (a, e, lambda_mean_ish, varpi) for a body relative to the central
    mass, in 2-D. lambda here is the true longitude (theta + varpi is folded in
    via the argument of periapsis); good enough to track a resonant argument."""
    x, y = pos[0], pos[1]
    vx, vy = vel[0], vel[1]
    r = math.hypot(x, y)
    v2 = vx * vx + vy * vy
    # semi-major axis from vis-viva
    a = 1.0 / (2.0 / r - v2 / mu)
    # eccentricity vector e = (v x h)/mu - r_hat  (2-D, h along z)
    h = x * vy - y * vx
    ex = (vy * h) / mu - x / r
    ey = (-vx * h) / mu - y / r
    e = math.hypot(ex, ey)
    varpi = math.atan2(ey, ex)          # longitude of periapsis
    theta = math.atan2(y, x)            # true longitude of the body
    return a, e, theta, varpi


def two_planet_system(a_inner: float, a_outer: float, m_star: float = 1.0,
                      m_inner: float = 1e-3, m_outer: float = 1e-3,
                      e_inner: float = 0.05, e_outer: float = 0.0,
                      G: float = G_DEFAULT) -> NBody:
    """Sun + two coplanar planets started at periapsis on the +x axis."""
    def planet(a, e, m):
        r_peri = a * (1.0 - e)
        mu = G * (m_star + m)
        v = math.sqrt(mu * (2.0 / r_peri - 1.0 / a))
        return [r_peri, 0.0, 0.0], [0.0, v, 0.0]

    p_in, v_in = planet(a_inner, e_inner, m_inner)
    p_out, v_out = planet(a_outer, e_outer, m_outer)
    masses = [m_star, m_inner, m_outer]
    pos = [[0.0, 0.0, 0.0], p_in, p_out]
    vel = [[0.0, 0.0, 0.0], v_in, v_out]
    body = NBody(masses=masses, pos=pos, vel=vel, G=G)
    # zero net momentum
    p = body.linear_momentum()
    for k in range(3):
        body.vel[0][k] -= p[k] / body.m[0]
    return body


def resonant_argument_series(system: NBody, p: int, q: int, dt: float,
                             steps: int, sample_every: int = 10):
    """Integrate and return (times, phi) where phi is the p:q resonant argument
    phi = p*theta_out - q*theta_in - (p-q)*varpi_in, wrapped to [-pi, pi]."""
    G = system.G
    mu_in = G * (system.m[0] + system.m[1])
    mu_out = G * (system.m[0] + system.m[2])
    ts, phis = [], []
    t = 0.0
    for s in range(steps):
        system.step("forest_ruth", dt)
        t += dt
        if s % sample_every == 0:
            _a1, _e1, th_in, varpi_in = _elements(system.pos[1], system.vel[1], mu_in)
            _a2, _e2, th_out, _vp2 = _elements(system.pos[2], system.vel[2], mu_out)
            phi = p * th_out - q * th_in - (p - q) * varpi_in
            # wrap to [-pi, pi]
            phi = (phi + math.pi) % (2.0 * math.pi) - math.pi
            ts.append(t); phis.append(phi)
    return ts, phis


def period_ratio(system: NBody, dt: float, steps: int) -> float:
    """Estimate T_outer / T_inner from the mean semi-major axes over a run
    (Kepler: T ~ a^{3/2})."""
    G = system.G
    mu_in = G * (system.m[0] + system.m[1])
    mu_out = G * (system.m[0] + system.m[2])
    a_in_sum = a_out_sum = 0.0
    n = 0
    for s in range(steps):
        system.step("forest_ruth", dt)
        if s % 10 == 0:
            a_in, *_ = _elements(system.pos[1], system.vel[1], mu_in)
            a_out, *_ = _elements(system.pos[2], system.vel[2], mu_out)
            a_in_sum += a_in; a_out_sum += a_out; n += 1
    a_in, a_out = a_in_sum / n, a_out_sum / n
    return (a_out / a_in) ** 1.5


def libration_amplitude(phis: List[float]) -> float:
    """Peak-to-peak range of phi. Small (< ~2pi and bounded away from full
    circulation) => librating/locked; ~2pi => circulating/not resonant."""
    return max(phis) - min(phis)

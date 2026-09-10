"""The Tisserand parameter: a near-invariant across a gravity assist.

When a small body (comet, asteroid, spacecraft) flies past a planet, the
encounter can drastically change its heliocentric orbit -- semi-major axis,
eccentricity, and inclination all shift. Yet one combination stays almost
unchanged:

    T_p = a_p/a + 2 sqrt( (a/a_p) (1 - e^2) ) cos i

This is the Tisserand parameter (relative to the planet's semi-major axis a_p).
It is essentially the Jacobi constant of the circular restricted three-body
problem re-expressed in the small body's osculating heliocentric elements, so it
is conserved to the same degree the CR3BP is a good model. Tisserand used it to
recognize returning comets whose orbits had been reshaped by Jupiter; it also
bounds what a single gravity assist can achieve and classifies small-body
populations (T_J > 3 asteroids, 2 < T_J < 3 Jupiter-family comets).

This module models a planar Sun + planet + massless body, integrates a flyby,
and shows a, e change a lot while T_p barely moves. Reuses NBody. Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from nbody import NBody

G_DEFAULT = 4.0 * math.pi ** 2  # AU, yr, solar masses


def tisserand(a: float, e: float, i_rad: float, a_p: float) -> float:
    """Tisserand parameter of an orbit (a, e, i) relative to planet radius a_p."""
    return a_p / a + 2.0 * math.sqrt((a / a_p) * (1.0 - e * e)) * math.cos(i_rad)


def heliocentric_elements(pos, vel, mu) -> Tuple[float, float, float]:
    """Osculating (a, e, i) of a body about the Sun. Planar model -> i is the
    angle of the orbit's angular-momentum vector from the z-axis (0 if in-plane;
    we keep the z terms so an inclined test works too)."""
    x, y, z = pos
    vx, vy, vz = vel
    r = math.sqrt(x * x + y * y + z * z)
    v2 = vx * vx + vy * vy + vz * vz
    a = 1.0 / (2.0 / r - v2 / mu)
    # angular momentum h = r x v
    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx
    h = math.sqrt(hx * hx + hy * hy + hz * hz)
    # eccentricity from e^2 = 1 - h^2/(mu a)
    e2 = max(0.0, 1.0 - h * h / (mu * a))
    e = math.sqrt(e2)
    inc = math.acos(max(-1.0, min(1.0, hz / h))) if h > 0 else 0.0
    return a, e, inc


def flyby(a0: float, e0: float, a_planet: float = 1.0, m_planet: float = 1e-3,
          m_star: float = 1.0, G: float = G_DEFAULT,
          dt: float = 5e-4, steps: int = 40000, sample_every: int = 50):
    """Integrate a massless body (started at aphelion) that crosses a planet's
    orbit and undergoes a flyby. Returns a list of (t, a, e, T_p) samples of the
    body's heliocentric elements and Tisserand parameter over the encounter."""
    mu_star = G * m_star
    # planet on a circular orbit
    v_planet = math.sqrt(G * m_star / a_planet)
    # test body: start at its aphelion on the -x axis so it swings in toward the
    # planet's orbit; give it (a0, e0)
    r_aph = a0 * (1.0 + e0)
    v_aph = math.sqrt(mu_star * (2.0 / r_aph - 1.0 / a0))
    masses = [m_star, m_planet, 0.0]
    pos = [[0.0, 0.0, 0.0], [a_planet, 0.0, 0.0], [-r_aph, 0.0, 0.0]]
    vel = [[0.0, 0.0, 0.0], [0.0, v_planet, 0.0], [0.0, -v_aph, 0.0]]
    body = NBody(masses=masses, pos=pos, vel=vel, G=G, softening=1e-3)
    # zero net momentum (Sun recoils)
    p = body.linear_momentum()
    for k in range(3):
        body.vel[0][k] -= p[k] / body.m[0]

    out = []
    t = 0.0
    for s in range(steps):
        body.step("forest_ruth", dt)
        t += dt
        if s % sample_every == 0:
            # elements relative to the Sun (index 0)
            rp = [body.pos[2][k] - body.pos[0][k] for k in range(3)]
            rv = [body.vel[2][k] - body.vel[0][k] for k in range(3)]
            a, e, inc = heliocentric_elements(rp, rv, mu_star)
            if a > 0:  # bound
                Tp = tisserand(a, e, inc, a_planet)
                out.append((t, a, e, Tp))
    return out

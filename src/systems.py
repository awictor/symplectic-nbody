"""Canonical N-body test systems with known behaviour."""

from __future__ import annotations

import math
from typing import Tuple, List

from nbody import NBody


def two_body_circular(m1: float = 1.0, m2: float = 1.0, r: float = 1.0) -> NBody:
    """Two equal masses on a circular orbit about their barycentre.
    With G=1 the exact orbital period is analytic, giving a clean test case."""
    mu = m1 + m2
    # circular relative speed for separation r: v_rel = sqrt(G*mu/r)
    v_rel = math.sqrt(mu / r)
    # place on x-axis, barycentre at origin
    x1 = -m2 / mu * r
    x2 = m1 / mu * r
    # velocities perpendicular, split by mass so total momentum is zero
    v1 = -m2 / mu * v_rel
    v2 = m1 / mu * v_rel
    return NBody(
        masses=[m1, m2],
        pos=[[x1, 0.0, 0.0], [x2, 0.0, 0.0]],
        vel=[[0.0, v1, 0.0], [0.0, v2, 0.0]],
    )


def two_body_eccentric(e: float = 0.7, m1: float = 1.0, m2: float = 1.0,
                       a: float = 1.0) -> NBody:
    """Two masses on an eccentric orbit (eccentricity e), started at apoapsis.
    The fast peri passage stresses fixed-step integrators, exposing RK4's
    secular energy drift while symplectic methods stay bounded."""
    mu = m1 + m2
    r_apo = a * (1.0 + e)
    # vis-viva at apoapsis: v_rel^2 = G*mu*(2/r - 1/a)
    v_rel = math.sqrt(mu * (2.0 / r_apo - 1.0 / a))
    x1 = -m2 / mu * r_apo
    x2 = m1 / mu * r_apo
    v1 = -m2 / mu * v_rel
    v2 = m1 / mu * v_rel
    return NBody(
        masses=[m1, m2],
        pos=[[x1, 0.0, 0.0], [x2, 0.0, 0.0]],
        vel=[[0.0, v1, 0.0], [0.0, v2, 0.0]],
    )


def figure_eight() -> NBody:
    """The Chenciner-Montgomery figure-eight: three equal masses chasing each
    other around a figure-8 orbit. A famous choreography solution; extremely
    sensitive to integrator quality, so it's a great stress test."""
    # Initial conditions (Chenciner & Montgomery 2000), G=1, m=1.
    x = 0.97000436
    y = -0.24308753
    vx = 0.4662036850
    vy = 0.4323657300
    return NBody(
        masses=[1.0, 1.0, 1.0],
        pos=[[x, y, 0.0], [-x, -y, 0.0], [0.0, 0.0, 0.0]],
        vel=[[vx, vy, 0.0], [vx, vy, 0.0], [-2 * vx, -2 * vy, 0.0]],
    )


def pythagorean() -> NBody:
    """Burrau's pythagorean three-body problem: masses 3,4,5 at rest on the
    vertices of a 3-4-5 right triangle. Chaotic, with multiple close encounters."""
    return NBody(
        masses=[3.0, 4.0, 5.0],
        pos=[[1.0, 3.0, 0.0], [-2.0, -1.0, 0.0], [1.0, -1.0, 0.0]],
        vel=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        softening=0.01,  # tame the singular close encounters
    )


def plummer_sphere(n: int = 500, seed: int = 1, total_mass: float = 1.0,
                   scale: float = 1.0) -> NBody:
    """A Plummer sphere: the standard equilibrium model of a star cluster.

    Positions follow the Plummer density profile; velocities are drawn from the
    exact isotropic distribution function via von Neumann rejection, so the cloud
    starts in near-virial equilibrium. Deterministic given `seed` (uses a small
    LCG so there are zero dependencies)."""
    rng = _LCG(seed)
    masses = [total_mass / n] * n
    pos: list = []
    vel: list = []
    for _ in range(n):
        # radius from inverse-CDF of the Plummer profile
        m = rng.uniform()
        r = scale / math.sqrt(m ** (-2.0 / 3.0) - 1.0)
        x, y, z = _random_unit_vector(rng)
        pos.append([r * x, r * y, r * z])

        # escape speed at r; sample q = v/v_esc from g(q) = q^2 (1-q^2)^{7/2}
        v_esc = math.sqrt(2.0) * (1.0 + r * r / (scale * scale)) ** (-0.25)
        while True:
            q = rng.uniform()
            g = q * q * (1.0 - q * q) ** 3.5
            if 0.1 * rng.uniform() < g:  # 0.1 >= max of g(q)
                break
        v = q * v_esc
        vx, vy, vz = _random_unit_vector(rng)
        vel.append([v * vx, v * vy, v * vz])

    return NBody(masses=masses, pos=pos, vel=vel, softening=scale / math.sqrt(n))


class _LCG:
    """Tiny deterministic PRNG (Numerical Recipes constants). No stdlib random,
    so results are reproducible across platforms and Python versions."""
    def __init__(self, seed: int):
        self.state = (seed * 2862933555777941757 + 1) & ((1 << 64) - 1)

    def uniform(self) -> float:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((self.state >> 11) / float(1 << 53)) or 1e-12


def _random_unit_vector(rng: "_LCG"):
    # Marsaglia's method for a uniform point on the unit sphere.
    while True:
        u = 2.0 * rng.uniform() - 1.0
        v = 2.0 * rng.uniform() - 1.0
        s = u * u + v * v
        if s < 1.0:
            break
    f = 2.0 * math.sqrt(1.0 - s)
    return u * f, v * f, 1.0 - 2.0 * s


SYSTEMS = {
    "two_body": two_body_circular,
    "two_body_eccentric": two_body_eccentric,
    "figure_eight": figure_eight,
    "pythagorean": pythagorean,
    "plummer": plummer_sphere,
}

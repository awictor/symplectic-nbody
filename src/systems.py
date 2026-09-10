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


SYSTEMS = {
    "two_body": two_body_circular,
    "two_body_eccentric": two_body_eccentric,
    "figure_eight": figure_eight,
    "pythagorean": pythagorean,
}

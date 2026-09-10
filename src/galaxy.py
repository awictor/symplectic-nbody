"""Disk galaxies and their collisions -- tidal tails from gravity alone.

A "galaxy" here is a heavy central point mass surrounded by a cold disk of light
tracer particles on near-circular orbits. Send two such galaxies past each other
and the differential tidal force draws out the bridges and tails seen in real
interacting galaxies (the Antennae, the Mice, ...). This is the classic
Toomre & Toomre (1972) restricted N-body picture.

Forces are computed with the project's Barnes-Hut tree so a few thousand
particles run in reasonable time. Deterministic (built-in LCG). Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from nbody import NBody
from systems import _LCG, _random_unit_vector  # reuse the dependency-free PRNG

Vec = List[float]


def _rotate_z(v: Vec, ang: float) -> Vec:
    c, s = math.cos(ang), math.sin(ang)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


def make_disk(center: Vec, bulk_vel: Vec, m_center: float, n_ring: int = 400,
              r_in: float = 0.3, r_out: float = 1.5, G: float = 1.0,
              seed: int = 1, sense: float = 1.0, inclination: float = 0.0):
    """Return (masses, positions, velocities) for a central mass plus a cold
    disk of `n_ring` massless tracers on circular orbits about it.

    sense = +/-1 sets the rotation direction; inclination tilts the disk about
    the x-axis so the encounter isn't perfectly coplanar."""
    rng = _LCG(seed)
    masses = [m_center]
    pos = [list(center)]
    vel = [list(bulk_vel)]

    inc = inclination
    ci, si = math.cos(inc), math.sin(inc)

    for _ in range(n_ring):
        # radius with uniform surface-density-ish sampling (sqrt for area weight)
        u = rng.uniform()
        r = math.sqrt(r_in * r_in + u * (r_out * r_out - r_in * r_in))
        phi = 2.0 * math.pi * rng.uniform()
        # position in the disk plane
        px, py = r * math.cos(phi), r * math.sin(phi)
        # circular speed about the central mass
        vc = math.sqrt(G * m_center / r) * sense
        vx, vy = -vc * math.sin(phi), vc * math.cos(phi)
        # tilt disk about x-axis by inclination
        p = [px, ci * py, si * py]
        v = [vx, ci * vy, si * vy]
        masses.append(0.0)  # massless tracer
        pos.append([center[0] + p[0], center[1] + p[1], center[2] + p[2]])
        vel.append([bulk_vel[0] + v[0], bulk_vel[1] + v[1], bulk_vel[2] + v[2]])

    return masses, pos, vel


def two_galaxy_encounter(n_ring: int = 400, m_center: float = 1.0,
                         b: float = 2.5, v_approach: float = 0.55,
                         sep: float = 6.0, G: float = 1.0,
                         inclination: float = 0.4) -> NBody:
    """Build two disk galaxies on a hyperbolic-ish passing encounter, offset by
    impact parameter `b`, approaching along x with speed `v_approach`."""
    # galaxy A: lower-left, moving +x ; galaxy B: upper-right, moving -x
    mA, pA, vA = make_disk(
        center=[-sep / 2, -b / 2, 0.0], bulk_vel=[v_approach, 0.0, 0.0],
        m_center=m_center, n_ring=n_ring, G=G, seed=11, sense=+1.0,
        inclination=inclination)
    mB, pB, vB = make_disk(
        center=[sep / 2, b / 2, 0.0], bulk_vel=[-v_approach, 0.0, 0.0],
        m_center=m_center, n_ring=n_ring, G=G, seed=23, sense=+1.0,
        inclination=-inclination)

    masses = mA + mB
    pos = pA + pB
    vel = vA + vB
    # soften on the scale of a tracer spacing to keep the tree well-behaved
    return NBody(masses=masses, pos=pos, vel=vel, G=G, softening=0.15)


def centers_of(system: NBody) -> Tuple[int, int]:
    """Indices of the two central masses (the only massive bodies)."""
    heavy = [i for i, m in enumerate(system.m) if m > 0.0]
    return heavy[0], heavy[1]

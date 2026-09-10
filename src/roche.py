"""Roche limit and tidal disruption of a rubble-pile satellite.

A satellite held together only by its own gravity survives as long as its
self-gravity beats the tidal force from the primary it orbits. Inside the Roche
limit the tide wins and the body is torn into a stream -- how Saturn's rings and
comet Shoemaker-Levy 9's fragment chain formed.

For two fluid/rubble bodies the classic Roche limit is

    d_Roche = 2.44 R_primary (rho_primary / rho_satellite)^{1/3}

We model the satellite as a small cluster of equal point masses (a "rubble
pile") bound by mutual gravity, place it on a circular orbit at distance d from a
heavy primary, integrate, and measure whether it stays bound or disperses. The
primary is a fixed point mass at the origin (the satellite is a test cluster in
its field, plus full self-gravity among the rubble particles).

Reuses NBody + the symplectic integrator + the built-in LCG. Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from nbody import NBody
from systems import _LCG


def roche_limit(R_primary: float, rho_primary: float, rho_sat: float) -> float:
    """Rigid/fluid Roche limit d = 2.44 R (rho_p / rho_s)^{1/3}."""
    return 2.44 * R_primary * (rho_primary / rho_sat) ** (1.0 / 3.0)


def _sphere_radius(mass: float, density: float) -> float:
    """Radius of a uniform sphere of given mass and density."""
    return (3.0 * mass / (4.0 * math.pi * density)) ** (1.0 / 3.0)


def make_rubble_satellite(d: float, m_primary: float, m_sat: float = 1e-6,
                          n: int = 30, sat_radius: float = 0.03, seed: int = 1,
                          G: float = 1.0) -> Tuple[NBody, int]:
    """A rubble-pile satellite of `n` equal particles filling a sphere of radius
    `sat_radius`, on a circular orbit of radius `d` about a primary of mass
    `m_primary` at the origin. Returns (system, i_primary).

    The particles carry a small isotropic velocity dispersion (~virial) so the
    cloud is not perfectly cold. What matters for the Roche test is comparative:
    at fixed initial conditions, particles are stripped far faster deeper inside
    the Roche limit, because the tidal field grows steeply as the orbit shrinks."""
    rng = _LCG(seed)
    masses = [m_primary]                 # index 0: the primary (fixed-ish, heavy)
    pos = [[0.0, 0.0, 0.0]]
    # circular orbital velocity of the satellite's centre about the primary
    v_orb = math.sqrt(G * m_primary / d)
    vel = [[0.0, 0.0, 0.0]]

    # virial velocity dispersion for the self-bound cloud: sigma ~ sqrt(G M / R)
    sigma = 0.4 * math.sqrt(G * m_sat / sat_radius)

    mp = m_sat / n
    pvel = []
    for _ in range(n):
        # uniform point in a sphere (rejection)
        while True:
            x = (2 * rng.uniform() - 1) * sat_radius
            y = (2 * rng.uniform() - 1) * sat_radius
            z = (2 * rng.uniform() - 1) * sat_radius
            if x * x + y * y + z * z <= sat_radius * sat_radius:
                break
        masses.append(mp)
        pos.append([d + x, y, z])
        # small random internal velocity (Gaussian-ish via summed uniforms)
        rv = [sigma * (rng.uniform() + rng.uniform() + rng.uniform() - 1.5) for _ in range(3)]
        pvel.append(rv)

    # remove net internal momentum so the cloud's centre moves at exactly v_orb
    cvx = sum(v[0] for v in pvel) / n
    cvy = sum(v[1] for v in pvel) / n
    cvz = sum(v[2] for v in pvel) / n
    for rv in pvel:
        vel.append([rv[0] - cvx, v_orb + rv[1] - cvy, rv[2] - cvz])

    # softening ~ inter-particle spacing keeps rubble self-gravity finite
    soft = sat_radius / n ** (1.0 / 3.0)
    return NBody(masses=masses, pos=pos, vel=vel, G=G, softening=soft), 0


def bound_fraction(system: NBody, i_primary: int = 0) -> float:
    """Fraction of satellite particles gravitationally bound to the satellite's
    own centroid (energy < 0 relative to the cloud). A bound satellite keeps this
    near 1; a tidally disrupted one sheds particles and it drops."""
    idx = [i for i in range(system.n) if i != i_primary]
    cx = [sum(system.pos[i][k] for i in idx) / len(idx) for k in range(3)]
    cv = [sum(system.vel[i][k] for i in idx) / len(idx) for k in range(3)]
    m_sat = sum(system.m[i] for i in idx)
    G = system.G
    nb = 0
    for i in idx:
        dr = math.sqrt(sum((system.pos[i][k] - cx[k]) ** 2 for k in range(3))) + 1e-12
        dv2 = sum((system.vel[i][k] - cv[k]) ** 2 for k in range(3))
        if 0.5 * dv2 - G * m_sat / dr < 0.0:
            nb += 1
    return nb / len(idx)


def satellite_spread(system: NBody, i_primary: int = 0) -> float:
    """RMS size of the satellite (spread of its particles about their centroid)."""
    idx = [i for i in range(system.n) if i != i_primary]
    cx = sum(system.pos[i][0] for i in idx) / len(idx)
    cy = sum(system.pos[i][1] for i in idx) / len(idx)
    cz = sum(system.pos[i][2] for i in idx) / len(idx)
    s = 0.0
    for i in idx:
        s += ((system.pos[i][0] - cx) ** 2 + (system.pos[i][1] - cy) ** 2
              + (system.pos[i][2] - cz) ** 2)
    return math.sqrt(s / len(idx))


def evolve_and_measure(system: NBody, i_primary: int, dt: float, steps: int
                       ) -> Tuple[float, float]:
    """Integrate; return (initial_spread, final_spread). A satellite that stays
    bound keeps roughly its initial spread; a disrupted one spreads far more."""
    s0 = satellite_spread(system, i_primary)
    for _ in range(steps):
        system.step("verlet", dt)
    return s0, satellite_spread(system, i_primary)


def surviving_bound_fraction(d_over_roche: float, R_primary: float = 0.3,
                             m_primary: float = 1.0, m_sat: float = 1e-3,
                             sat_radius: float = 0.05, n: int = 60,
                             seed: int = 4, dt: float = 0.002, steps: int = 1500,
                             G: float = 1.0) -> float:
    """Convenience: build a rubble satellite at `d_over_roche` times the Roche
    limit, evolve for a fixed time, and return the bound fraction that survives.
    Keeps the (mass, radius, timing) fixed so results at different distances are
    directly comparable -- isolating the tidal effect."""
    rho_sat = m_sat / (4.0 / 3.0 * math.pi * sat_radius ** 3)
    rho_p = m_primary / (4.0 / 3.0 * math.pi * R_primary ** 3)
    d = d_over_roche * roche_limit(R_primary, rho_p, rho_sat)
    system, ip = make_rubble_satellite(d, m_primary, m_sat=m_sat, n=n,
                                       sat_radius=sat_radius, seed=seed, G=G)
    for _ in range(steps):
        system.step("verlet", dt)
    return bound_fraction(system, ip)

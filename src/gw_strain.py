"""Gravitational-wave strain: the number LIGO measures.

A gravitational wave stretches and squeezes space by a fractional amount h -- the
STRAIN. For an inspiralling binary of chirp mass M_c at distance d emitting at
gravitational-wave frequency f, the amplitude is

    h ~ (4 / d) (G M_c / c^2)^{5/3} (pi f / c)^{2/3},

where the chirp mass M_c = (m1 m2)^{3/5} / (m1 + m2)^{1/5} sets both the amplitude
and the chirp rate. For GW150914 (two ~30 solar-mass black holes ~410 Mpc away,
merging near f_gw ~ 250 Hz) this comes out to h ~ 1e-21 -- the fantastically tiny
number LIGO detected. Over LIGO's 4 km arms that is a length change of

    Delta L = h L ~ 1e-21 * 4000 m ~ 4e-18 m,

a thousandth the width of a proton. This module computes the chirp mass, the
strain, and the arm-length change, and reproduces the GW150914 figures. SI units.
Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
M_SUN = 1.98892e30
MPC = 3.0857e22                # megaparsec, m
LIGO_ARM = 4000.0              # m


def chirp_mass(m1: float, m2: float) -> float:
    """Chirp mass M_c = (m1 m2)^{3/5} / (m1 + m2)^{1/5}."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def strain(m1: float, m2: float, distance: float, f_gw: float) -> float:
    """Order-of-magnitude GW strain amplitude
    h = (4/d)(G M_c/c^2)^{5/3}(pi f/c)^{2/3}."""
    Mc = chirp_mass(m1, m2)
    return (4.0 / distance) * (G * Mc / C ** 2) ** (5.0 / 3.0) \
        * (math.pi * f_gw / C) ** (2.0 / 3.0)


def arm_length_change(h: float, arm_length: float = LIGO_ARM) -> float:
    """Physical length change Delta L = h L of a detector arm."""
    return h * arm_length


def gw150914_strain() -> float:
    """GW150914-like strain: 36 + 29 solar masses at 410 Mpc, f_gw ~ 150 Hz."""
    return strain(36.0 * M_SUN, 29.0 * M_SUN, 410.0 * MPC, 150.0)

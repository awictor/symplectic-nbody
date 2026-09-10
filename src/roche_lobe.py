"""Roche lobes and binary mass transfer: how close binaries feed each other.

In a binary, each star is surrounded by its ROCHE LOBE -- the teardrop region
within which gas is gravitationally bound to that star, meeting the companion's
lobe at the inner Lagrange point L1. When a star swells (evolving off the main
sequence) or the orbit shrinks until it fills its lobe, gas spills through L1
onto the companion. That mass transfer powers cataclysmic variables, X-ray
binaries, novae, and type-Ia supernovae.

Eggleton (1983) gives an accurate fit for the Roche-lobe radius of the star of
mass ratio q = M1/M2:

    R_L / a = 0.49 q^{2/3} / ( 0.6 q^{2/3} + ln(1 + q^{1/3}) ).

Whether transfer is STABLE depends on how the donor's Roche lobe changes as it
loses mass versus how the donor itself responds: conservative transfer from a
less massive donor widens the orbit (stable), while from a more massive donor it
shrinks the orbit and can run away (unstable). This module gives the Eggleton
radius, the L1 distance, and the mass-transfer stability criterion. Units of the
orbital separation a. Pure stdlib.
"""

from __future__ import annotations

import math


def eggleton_radius(q: float) -> float:
    """Roche-lobe radius R_L/a for a star of mass ratio q = M_this / M_companion
    (Eggleton 1983)."""
    q13 = q ** (1.0 / 3.0)
    q23 = q ** (2.0 / 3.0)
    return 0.49 * q23 / (0.6 * q23 + math.log(1.0 + q13))


def l1_distance(q: float) -> float:
    """Approximate distance of L1 from the more massive star (mass ratio q =
    M2/M1 <= 1), as a fraction of separation: the classic series
    x_L1/a ~ 0.5 - 0.227 log10(q) is a decent fit for 0.1 < q < 10.
    Returned relative to the primary."""
    return 0.5 - 0.227 * math.log10(q)


def conservative_orbit_response(q: float) -> float:
    """d ln a / d ln M_donor for CONSERVATIVE mass transfer (total mass and
    angular momentum fixed), with q = M_donor / M_accretor:

        d ln a / d ln M_d = 2 (q - 1).

    Negative (orbit widens as donor loses mass) when q < 1 -> stable; positive
    (orbit shrinks) when q > 1 -> tends to unstable runaway."""
    return 2.0 * (q - 1.0)


def transfer_is_stable(q: float) -> bool:
    """Conservative mass transfer is dynamically stable when the donor is the
    LESS massive star (q = M_donor/M_accretor < 1): losing mass widens the orbit,
    detaching the donor. From a more massive donor it shrinks and runs away."""
    return q < 1.0

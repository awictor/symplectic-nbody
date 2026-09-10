"""Two-body relaxation and cluster evaporation: how star clusters slowly forget.

A star crossing a cluster is deflected a little by every other star it passes. Any one
encounter is tiny, but the accumulated random kicks eventually change the star's
velocity by order itself -- the star "forgets" its original orbit. The time for that
is the two-body relaxation time,

    t_relax ~ (N / (8 ln N)) * t_cross,

with N the number of stars and t_cross = R/v the crossing time. The Coulomb logarithm
ln(N) (really ln of the ratio of maximum to minimum impact parameter) counts how the
many distant, weak encounters dominate over the few close ones.

Because t_relax grows almost linearly with N, big systems relax slowly: a globular
cluster (N ~ 1e5, t_cross ~ 1 Myr) relaxes in ~100 Myr-1 Gyr and has had time to reach
a relaxed, mass-segregated state within a Hubble time, whereas a galaxy (N ~ 1e11)
has a relaxation time far longer than the age of the universe -- it is effectively
collisionless, which is why galaxies keep their spiral arms and streams.

Relaxation drives evaporation: two-body kicks occasionally push a star above the
escape speed, and it leaves. A roughly constant ~1% of stars evaporate per relaxation
time, so the cluster slowly bleeds stars and its core contracts, giving an evaporation
(dissolution) time of order t_evap ~ 100 t_relax.

This module gives the crossing time, the relaxation time, the collisionless verdict,
and the evaporation time, and reproduces the globular-relaxed / galaxy-collisionless
contrast. SI units. Pure stdlib; the collisional-dynamics companion to the virial and
cluster modules.
"""

from __future__ import annotations

import math

G = 6.67430e-11
PC = 3.0856775814913673e16
M_SUN = 1.989e30
KM = 1e3
MYR = 3.15576e13
GYR = 1e9 * 3.15576e7
HUBBLE_TIME = 1.38e10 * 3.15576e7   # ~13.8 Gyr in seconds

# fraction of stars lost per relaxation time (evaporation), t_evap ~ t_relax / this
EVAP_FRACTION = 0.0074              # ~ classic Ambartsumian/Spitzer value


def crossing_time(R: float, v: float) -> float:
    """Crossing time t_cross = R / v (s): how long a star takes to traverse the
    system once."""
    return R / v


def relaxation_time(N: int, t_cross: float) -> float:
    """Two-body relaxation time t_relax = (N / (8 ln N)) t_cross (s). The Coulomb
    logarithm ln N encodes the dominance of many weak, distant encounters."""
    return N / (8.0 * math.log(N)) * t_cross


def relaxation_time_from_params(N: int, R: float, v: float) -> float:
    """Relaxation time built directly from N, size R and velocity dispersion v."""
    return relaxation_time(N, crossing_time(R, v))


def is_collisionless(N: int, t_cross: float,
                     age: float = HUBBLE_TIME) -> bool:
    """True if the system is effectively collisionless: its relaxation time exceeds
    the given age (default a Hubble time). Galaxies are collisionless; open clusters
    are not."""
    return relaxation_time(N, t_cross) > age


def evaporation_time(N: int, t_cross: float,
                     frac: float = EVAP_FRACTION) -> float:
    """Cluster evaporation (dissolution) time t_evap = t_relax / frac (s), for a
    roughly constant fraction `frac` of stars escaping per relaxation time."""
    return relaxation_time(N, t_cross) / frac

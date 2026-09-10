"""The habitable zone: where a planet can hold liquid water.

A planet's equilibrium temperature -- the blackbody temperature at which it
re-radiates the starlight it absorbs -- is

    T_eq = T_star (R_star / 2d)^{1/2} (1 - A)^{1/4},

with d the orbital distance and A the Bode albedo. Setting T_eq to the edges of
the liquid-water range (roughly 273-373 K for the simple equilibrium estimate,
or the commonly used ~climate bounds) gives the HABITABLE ZONE: the band of
orbits where surface water could persist.

Because T_eq ~ (L_star)^{1/4} / d^{1/2}, the zone's distance scales as
sqrt(L_star): a luminous star's habitable zone sits far out, a dim red dwarf's is
tucked in close (where tidal locking bites). This module computes the equilibrium
temperature and the HZ bounds, and reproduces Earth's ~255 K equilibrium
temperature (the greenhouse effect warms the actual surface to 288 K).

Distances in AU, luminosities in solar units. Pure stdlib; complements
main_sequence and blackbody.
"""

from __future__ import annotations

import math

T_SUN = 5772.0
R_SUN = 6.957e8
AU = 1.495978707e11


def equilibrium_temperature(L_star_lsun: float, d_au: float,
                            albedo: float = 0.3) -> float:
    """Planet equilibrium temperature (K). Using L in solar units and d in AU,
    T_eq = 278.5 K * (1 - A)^{1/4} * L^{1/4} / sqrt(d)  (278.5 K is Earth's
    zero-albedo value at 1 AU from the Sun)."""
    return 278.5 * (1.0 - albedo) ** 0.25 * L_star_lsun ** 0.25 / math.sqrt(d_au)


def habitable_zone(L_star_lsun: float, T_inner: float = 273.0,
                   T_outer: float = 373.0, albedo: float = 0.3):
    """Inner and outer HZ radii (AU) where the equilibrium temperature crosses
    the given liquid-water bounds. Since T ~ 1/sqrt(d), invert for d."""
    # d = (278.5 (1-A)^1/4 L^1/4 / T)^2
    def d_for_T(T):
        return (278.5 * (1.0 - albedo) ** 0.25 * L_star_lsun ** 0.25 / T) ** 2

    # hotter bound -> inner edge, cooler bound -> outer edge
    return d_for_T(T_outer), d_for_T(T_inner)


def hz_center(L_star_lsun: float, albedo: float = 0.3) -> float:
    """Center of the HZ, which scales as sqrt(L_star)."""
    inner, outer = habitable_zone(L_star_lsun, albedo=albedo)
    return 0.5 * (inner + outer)

"""The main sequence: mass-luminosity relation and stellar lifetimes.

Most stars spend most of their lives fusing hydrogen on the "main sequence,"
where luminosity, radius, temperature, and lifetime are all set mainly by mass.
The empirical mass-luminosity relation is a steep power law,

    L / L_sun ~ (M / M_sun)^{3.5},

so a star's fuel (proportional to M) is burned at a rate that rises much faster
than the fuel supply. Its main-sequence lifetime therefore falls steeply with
mass:

    t_MS / t_sun ~ (M/M_sun) / (L/L_sun) ~ (M/M_sun)^{-2.5}.

The Sun (1 M_sun, 1 L_sun) lives ~10 Gyr; a 10 M_sun star lives only ~30 Myr; a
0.3 M_sun red dwarf outlives the current universe many times over. Coupling
L = 4 pi R^2 sigma T^4 lets us place stars on the Hertzsprung-Russell diagram.

This module gives the mass-luminosity relation, radius/temperature scalings, and
main-sequence lifetimes, in solar units. Pure stdlib.
"""

from __future__ import annotations

import math

T_SUN = 1.0e10 / 1.0            # placeholder; we work in Gyr below
SIGMA_SB = 5.670374419e-8
L_SUN = 3.828e26                # W
R_SUN = 6.957e8                 # m
T_SUN_EFF = 5772.0             # K
T_MS_SUN_GYR = 10.0            # Sun's main-sequence lifetime, Gyr


def luminosity(mass_msun: float) -> float:
    """Main-sequence luminosity in solar luminosities: L ~ M^{3.5}."""
    return mass_msun ** 3.5


def lifetime_gyr(mass_msun: float) -> float:
    """Main-sequence lifetime in Gyr: t ~ (M/L) t_sun ~ M^{-2.5} * 10 Gyr."""
    return T_MS_SUN_GYR * mass_msun / luminosity(mass_msun)


def radius_msun(mass_msun: float) -> float:
    """Approximate main-sequence radius in solar radii: R ~ M^{0.8}
    (a rough fit across the lower and upper main sequence)."""
    return mass_msun ** 0.8


def effective_temperature(mass_msun: float) -> float:
    """Effective temperature (K) from L = 4 pi R^2 sigma T^4:
    T = T_sun (L/L_sun)^{1/4} (R/R_sun)^{-1/2}."""
    L = luminosity(mass_msun)
    R = radius_msun(mass_msun)
    return T_SUN_EFF * (L ** 0.25) * (R ** -0.5)


def luminosity_watts(mass_msun: float) -> float:
    """Luminosity in watts."""
    return luminosity(mass_msun) * L_SUN

"""The Eddington luminosity: how bright an accreting object can be.

Radiation carries momentum, so a luminous object pushes outward on the ionized
gas around it (via Thomson scattering off electrons, which drag the protons along
by electrostatic coupling). When that radiation force balances gravity, accretion
shuts itself off. The balance point is the Eddington luminosity

    L_Edd = 4 pi G M m_p c / sigma_T,

independent of radius and linear in mass: L_Edd ~ 1.26e31 (M / M_sun) watts.
Brighter than this and the object blows its own fuel away. It sets:
  * the maximum steady accretion rate  Mdot_Edd = L_Edd / (eta c^2)  (eta ~ 0.1),
  * the Salpeter e-folding time ~45 Myr on which a black hole can grow, which
    bounds how fast the first quasars' billion-solar-mass holes could form.

This module gives L_Edd, the Eddington accretion rate and temperature scale, and
the Salpeter growth time, and checks the solar value. SI units. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
M_P = 1.67262192e-27           # proton mass
SIGMA_T = 6.6524587e-29        # Thomson cross-section, m^2
M_SUN = 1.98892e30
L_SUN = 3.828e26               # watts
YEAR = 3.15576e7


def eddington_luminosity(M: float) -> float:
    """L_Edd = 4 pi G M m_p c / sigma_T, in watts."""
    return 4.0 * math.pi * G * M * M_P * C / SIGMA_T


def eddington_luminosity_solar_units(M_in_msun: float) -> float:
    """L_Edd in solar luminosities for a mass in solar masses."""
    return eddington_luminosity(M_in_msun * M_SUN) / L_SUN


def eddington_accretion_rate(M: float, efficiency: float = 0.1) -> float:
    """Maximum steady accretion rate Mdot = L_Edd / (eta c^2), in kg/s."""
    return eddington_luminosity(M) / (efficiency * C * C)


def salpeter_time(efficiency: float = 0.1, f_edd: float = 1.0) -> float:
    """Salpeter e-folding growth time of an Eddington-limited accretor,
    t_S = eta sigma_T c / (4 pi G m_p f_edd). Independent of mass; ~45 Myr for
    eta=0.1 and f_edd=1. Returns seconds."""
    return efficiency * SIGMA_T * C / (4.0 * math.pi * G * M_P * f_edd)


def growth_time(M_initial: float, M_final: float, efficiency: float = 0.1,
                f_edd: float = 1.0) -> float:
    """Time to grow from M_initial to M_final under Eddington-limited accretion,
    t = t_S ln(M_final / M_initial). Seconds."""
    return salpeter_time(efficiency, f_edd) * math.log(M_final / M_initial)

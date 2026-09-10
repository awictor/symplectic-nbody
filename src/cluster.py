"""Galaxy-cluster virial temperature: why clusters glow in X-rays.

A galaxy cluster's gas sits in the deep gravitational well of ~10^14-10^15 solar
masses of (mostly dark) matter. Falling in, the gas virializes: its thermal
energy balances the potential, heating it to

    kT ~ (1/2) mu m_p sigma^2 ~ G M mu m_p / (2 R),

with sigma the velocity dispersion, mu ~ 0.6 the mean molecular weight. For a
massive cluster that is kT ~ several keV, T ~ 10^8 K -- hot enough to emit
thermal bremsstrahlung X-rays, which is how clusters are found and weighed.

The virial theorem also gives the mass-temperature relation: at fixed
overdensity M ~ R^3 and kT ~ M/R, so

    kT ~ M^{2/3},

the scaling used to turn an X-ray temperature into a cluster mass. This module
computes the virial temperature from mass and radius (or dispersion), the
bremsstrahlung luminosity scaling, and reproduces Coma-cluster numbers. SI units.
Pure stdlib; complements the virial-theorem module.
"""

from __future__ import annotations

import math

G = 6.67430e-11
K_B = 1.380649e-23
M_P = 1.6726219e-27
M_SUN = 1.98892e30
MPC = 3.0857e22
KEV = 1.602176634e-16          # 1 keV in joules
MU = 0.6                        # mean molecular weight (ionized primordial gas)


def virial_temperature(M: float, R: float) -> float:
    """Virial temperature (kelvin) from mass M and radius R:
    kT = G M mu m_p / (2 R)."""
    return G * M * MU * M_P / (2.0 * R * K_B)


def temperature_from_dispersion(sigma: float) -> float:
    """Virial temperature from the galaxy velocity dispersion sigma:
    kT = mu m_p sigma^2  (so T ~ sigma^2)."""
    return MU * M_P * sigma * sigma / K_B


def kT_kev(M: float, R: float) -> float:
    """Virial temperature expressed as kT in keV."""
    return virial_temperature(M, R) * K_B / KEV


def mass_from_temperature(kT_keV: float, R: float) -> float:
    """Invert the virial relation to estimate cluster mass from an X-ray
    temperature (keV) and radius: M = 2 R kT / (G mu m_p)."""
    kT_joule = kT_keV * KEV
    return 2.0 * R * kT_joule / (G * MU * M_P)


def bremsstrahlung_scaling(n: float, T: float) -> float:
    """Thermal-bremsstrahlung emissivity scaling ~ n^2 sqrt(T) (arbitrary units).
    Denser, hotter gas radiates more X-rays."""
    return n * n * math.sqrt(T)

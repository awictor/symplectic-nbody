"""Cosmological distances and the discovery of cosmic acceleration.

Once you have the expansion history (friedmann.Cosmology), every distance in
cosmology follows from a single integral of 1/E(z) over redshift:

    comoving distance     D_C(z) = (c/H0) integral_0^z dz' / E(z')
    luminosity distance   D_L(z) = (1+z) D_C(z)          (flat universe)
    angular-diameter dist D_A(z) = D_C(z) / (1+z)
    distance modulus      mu(z)  = 5 log10( D_L / 10 pc )

Two famous consequences:

  * a universe with dark energy expands faster late, so a given redshift lies at
    a LARGER luminosity distance -- distant type-Ia supernovae look FAINTER than
    a decelerating universe predicts. That faintness (Riess/Perlmutter 1998) is
    the evidence for cosmic acceleration and the 2011 Nobel Prize.
  * D_A(z) is non-monotonic: it rises, peaks (around z ~ 1.5-1.6 for LCDM), then
    FALLS, so the most distant objects can look angularly larger -- the reason
    the cosmic microwave background's spots subtend ~1 degree.

Distances are returned in units of the Hubble distance c/H0. Pure stdlib; reuses
friedmann.Cosmology for E(z).
"""

from __future__ import annotations

import math
from typing import List

from friedmann import Cosmology


def _E_of_z(cosmo: Cosmology, z: float) -> float:
    """Dimensionless Hubble rate as a function of redshift, E(z) = H(z)/H0.
    Uses a = 1/(1+z)."""
    return cosmo.E(1.0 / (1.0 + z))


def comoving_distance(cosmo: Cosmology, z: float, n: int = 20000) -> float:
    """D_C(z)/(c/H0) = integral_0^z dz'/E(z'), by Simpson's rule."""
    if z <= 0.0:
        return 0.0
    h = z / n
    total = 1.0 / _E_of_z(cosmo, 0.0) + 1.0 / _E_of_z(cosmo, z)
    for i in range(1, n):
        zi = i * h
        total += (4.0 if i % 2 == 1 else 2.0) / _E_of_z(cosmo, zi)
    return total * h / 3.0


def luminosity_distance(cosmo: Cosmology, z: float) -> float:
    """D_L(z)/(c/H0) = (1+z) D_C(z) for a flat universe."""
    return (1.0 + z) * comoving_distance(cosmo, z)


def angular_diameter_distance(cosmo: Cosmology, z: float) -> float:
    """D_A(z)/(c/H0) = D_C(z)/(1+z) for a flat universe."""
    return comoving_distance(cosmo, z) / (1.0 + z)


def distance_modulus(cosmo: Cosmology, z: float, H0_km_s_Mpc: float = 70.0) -> float:
    """mu(z) = 5 log10(D_L / 10 pc). Converts the dimensionless D_L to physical
    units using the Hubble distance c/H0 (Mpc) for the given H0."""
    c_km_s = 299792.458
    hubble_dist_mpc = c_km_s / H0_km_s_Mpc         # c/H0 in Mpc
    D_L_mpc = luminosity_distance(cosmo, z) * hubble_dist_mpc
    D_L_pc = D_L_mpc * 1e6
    return 5.0 * math.log10(D_L_pc / 10.0)


def angular_diameter_peak(cosmo: Cosmology, z_max: float = 5.0,
                          n: int = 500) -> float:
    """Redshift at which the angular-diameter distance peaks (the turnover)."""
    best_z, best_d = 0.0, -1.0
    for i in range(1, n + 1):
        z = z_max * i / n
        d = angular_diameter_distance(cosmo, z)
        if d > best_d:
            best_d, best_z = d, z
    return best_z

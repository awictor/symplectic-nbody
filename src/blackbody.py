"""Blackbody radiation: Planck's law, Wien's peak, Stefan-Boltzmann.

A body in thermal equilibrium radiates the universal Planck spectrum, whose shape
depends only on temperature. Three results follow:

  * PLANCK's law for spectral radiance vs wavelength,
        B(lambda, T) = (2 h c^2 / lambda^5) / (exp(hc / lambda k T) - 1);
  * WIEN's displacement law -- the peak wavelength times temperature is constant,
        lambda_max T = b = 2.898e-3 m K   (so hot things are blue, cool things red);
  * the STEFAN-BOLTZMANN law -- total emitted power per area goes as T^4,
        j = sigma T^4.

These set the color of stars, the ~500 nm peak of the Sun, the peak of the 2.725
K cosmic microwave background in the microwave, and how luminosity scales with
temperature. This module gives the Planck spectrum, Wien peak, and Stefan-
Boltzmann flux, and reproduces those numbers. SI units. Pure stdlib.
"""

from __future__ import annotations

import math

H = 6.62607015e-34
C = 2.99792458e8
K_B = 1.380649e-23
SIGMA_SB = 5.670374419e-8       # Stefan-Boltzmann constant, W m^-2 K^-4
WIEN_B = 2.897771955e-3         # Wien displacement constant, m K


def planck_wavelength(lam: float, T: float) -> float:
    """Planck spectral radiance B(lambda, T) in W sr^-1 m^-3."""
    x = H * C / (lam * K_B * T)
    if x > 700.0:               # exp overflows; radiance is negligibly small here
        return 0.0
    return (2.0 * H * C ** 2 / lam ** 5) / (math.expm1(x))


def wien_peak_wavelength(T: float) -> float:
    """Peak wavelength of the Planck spectrum: lambda_max = b / T."""
    return WIEN_B / T


def stefan_boltzmann_flux(T: float) -> float:
    """Total radiated power per unit area, j = sigma T^4 (W/m^2)."""
    return SIGMA_SB * T ** 4


def luminosity(R: float, T: float) -> float:
    """Luminosity of a sphere of radius R at temperature T: L = 4 pi R^2 sigma T^4."""
    return 4.0 * math.pi * R * R * stefan_boltzmann_flux(T)


def peak_wavelength_numeric(T: float, lam_lo: float = 1e-9,
                            lam_hi: float = 1e-2, n: int = 20000) -> float:
    """Find the Planck peak wavelength by scanning -- a cross-check on Wien."""
    best_lam, best_B = lam_lo, -1.0
    for i in range(n):
        lam = lam_lo * (lam_hi / lam_lo) ** (i / (n - 1))
        B = planck_wavelength(lam, T)
        if B > best_B:
            best_B, best_lam = B, lam
    return best_lam

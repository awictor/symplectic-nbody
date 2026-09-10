"""Stellar opacity: how hard it is for light to escape a star.

Opacity kappa (m^2/kg) measures how strongly matter absorbs and scatters radiation:
the photon mean free path is 1/(kappa rho), and it is opacity that sets how slowly a
star leaks its luminosity and therefore how it is structured. Several processes
dominate in different regimes.

Electron (Thomson) scattering is temperature- and density-independent,

    kappa_es = sigma_T / m_p * (1 + X) / 2  ~  0.02 (1 + X) m^2/kg,

the floor opacity in hot, ionized, low-density gas (X is the hydrogen mass fraction).
It is what the Eddington luminosity is built on.

Bound-free and free-free absorption follow Kramers' law, a steep power law that peaks
at intermediate temperatures,

    kappa_Kramers = kappa_0 rho T^(-7/2),

rising with density and falling as T^(-3.5) -- so the cooler outer layers of a star
are far more opaque than the blazing core. H-minus opacity dominates the cool (~few
thousand K) envelopes of Sun-like stars, where a stray electron on a hydrogen atom
absorbs voraciously, rising steeply with temperature until hydrogen ionizes.

Which process wins sets whether energy moves by radiation or convection, and the
total is roughly the sum of the channels. This module gives the electron-scattering,
Kramers bound-free/free-free, and H-minus opacities and their sum, and reproduces the
Thomson floor and the T^(-3.5) Kramers scaling. SI units. Pure stdlib; the
opacity that underlies the Eddington-limit and main-sequence modules.
"""

from __future__ import annotations

import math

SIGMA_T = 6.6524587e-29        # Thomson cross section (m^2)
M_P = 1.6726219e-27            # proton mass (kg)

# Kramers coefficients (SI: rho in kg/m^3, T in K, kappa in m^2/kg), calibrated so
# free-free gives ~0.07 m^2/kg (~0.7 cm^2/g) at the solar centre.
KAPPA0_BF = 4.3e18             # bound-free Kramers coefficient
KAPPA0_FF = 3.7e18             # free-free Kramers coefficient


def electron_scattering(X: float = 0.7) -> float:
    """Thomson electron-scattering opacity (m^2/kg): kappa = sigma_T/m_p (1+X)/2.
    Temperature- and density-independent; the floor in hot ionized gas."""
    return SIGMA_T / M_P * (1.0 + X) / 2.0


def kramers_bound_free(rho: float, T: float, X: float = 0.7,
                       Z: float = 0.02) -> float:
    """Bound-free Kramers opacity (m^2/kg): kappa ~ kappa0 Z(1+X) rho T^(-7/2).
    Dominates where metals are partially ionized."""
    return KAPPA0_BF * Z * (1.0 + X) * rho * T ** (-3.5)


def kramers_free_free(rho: float, T: float, X: float = 0.7) -> float:
    """Free-free Kramers opacity (m^2/kg): kappa ~ kappa0 (X+Y)(1+X) rho T^(-7/2)."""
    return KAPPA0_FF * (1.0 + X) * rho * T ** (-3.5)


def h_minus(rho: float, T: float, Z: float = 0.02) -> float:
    """H-minus opacity (m^2/kg), a steep low-temperature term ~ Z rho^(1/2) T^9,
    dominant in the ~3000-6000 K envelopes of cool stars (valid only there)."""
    return 1.1e-31 * (Z / 0.02) * rho ** 0.5 * T ** 9.0


def total_opacity(rho: float, T: float, X: float = 0.7, Z: float = 0.02) -> float:
    """Approximate total opacity: electron scattering plus Kramers bound-free and
    free-free (the hot-interior channels), summed."""
    return (electron_scattering(X)
            + kramers_bound_free(rho, T, X, Z)
            + kramers_free_free(rho, T, X))


def mean_free_path(rho: float, kappa: float) -> float:
    """Photon mean free path 1/(kappa rho) (m)."""
    return 1.0 / (kappa * rho)

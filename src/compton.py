"""Compton and inverse-Compton scattering: photons trading energy with electrons.

When a photon scatters off a free electron its wavelength lengthens by the
Compton shift,

    Delta lambda = lambda_C (1 - cos theta),   lambda_C = h / (m_e c) = 2.426 pm,

the classic experiment (Compton 1923) that proved light carries momentum like a
particle. The shift is largest for back-scattering (theta = 180 deg) and zero for
forward scattering.

INVERSE Compton scattering runs it backwards: a fast electron kicks a low-energy
photon UP in energy by a factor ~ gamma^2, turning starlight or CMB photons into
X-rays and gamma-rays -- the process behind the Sunyaev-Zeldovich effect and
much of high-energy astrophysics. The Thomson cross-section sigma_T sets the
scattering rate.

This module gives the Compton shift, the scattered photon energy, the Compton
wavelength, and the inverse-Compton boost, and reproduces the textbook numbers.
SI units. Pure stdlib.
"""

from __future__ import annotations

import math

H = 6.62607015e-34
C = 2.99792458e8
M_E = 9.1093837e-31
EV = 1.602176634e-19
KEV = 1e3 * EV
SIGMA_T = 6.6524587e-29        # Thomson cross-section, m^2

LAMBDA_C = H / (M_E * C)       # Compton wavelength, ~2.426e-12 m


def compton_shift(theta_rad: float) -> float:
    """Wavelength increase Delta lambda = lambda_C (1 - cos theta) (m)."""
    return LAMBDA_C * (1.0 - math.cos(theta_rad))


def scattered_wavelength(lam0: float, theta_rad: float) -> float:
    """Scattered-photon wavelength lambda' = lambda0 + Delta lambda."""
    return lam0 + compton_shift(theta_rad)


def scattered_energy(E0_joule: float, theta_rad: float) -> float:
    """Scattered-photon energy from the Compton formula:
    E' = E0 / (1 + (E0 / m_e c^2)(1 - cos theta))."""
    mec2 = M_E * C * C
    return E0_joule / (1.0 + (E0_joule / mec2) * (1.0 - math.cos(theta_rad)))


def inverse_compton_boost(gamma: float) -> float:
    """Mean energy-boost factor ~ (4/3) gamma^2 beta^2 for an isotropic photon
    field scattered by relativistic electrons of Lorentz factor gamma."""
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    return (4.0 / 3.0) * gamma * gamma * beta2


def electron_rest_energy_kev() -> float:
    """m_e c^2 in keV (~511 keV)."""
    return M_E * C * C / KEV

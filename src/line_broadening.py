"""Spectral line broadening: why atomic lines have width.

A spectral line is never infinitely sharp. Three mechanisms give it width, and their
relative sizes read out the physics of the emitting gas.

DOPPLER (thermal) broadening. Atoms move with a Maxwell-Boltzmann velocity spread, so
their emission is red- and blue-shifted by +/- v/c. This gives a Gaussian profile of
1/e half-width in frequency

    delta_nu_D = (nu0 / c) sqrt(2 k_B T / m),

widening with temperature and favouring light atoms. It dominates in hot, thin gas
(stellar photospheres, HII regions) and is the standard thermometer of an emitting
plasma.

NATURAL (lifetime) broadening. The excited state lives only ~1/A seconds (A the
Einstein coefficient), so by the uncertainty principle the line has an irreducible
Lorentzian width

    delta_nu_nat = A / (4 pi).

PRESSURE (collisional) broadening. Collisions at rate nu_col interrupt the emission,
adding a Lorentzian width delta_nu_p = nu_col / (2 pi) that grows with density -- which
is why dwarf-star lines (high photospheric pressure) are broader than giant-star lines
and lets line width diagnose surface gravity.

The observed profile is the convolution of the Gaussian (Doppler) and Lorentzian
(natural + pressure) parts -- a Voigt profile. This module gives each width, the
thermal velocity, and the dominant mechanism, and reproduces the Doppler-dominated
hot-gas line and the pressure-broadened dwarf line. SI units. Pure stdlib; the
line-shape companion to the Saha and blackbody modules.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
C = 2.99792458e8
AMU = 1.66053907e-27


def thermal_speed(T: float, m: float) -> float:
    """Most-probable thermal speed sqrt(2 k_B T / m) (m/s)."""
    return math.sqrt(2.0 * K_B * T / m)


def doppler_width(nu0: float, T: float, m: float) -> float:
    """Doppler (thermal) 1/e half-width in frequency (Hz):
    delta_nu = (nu0/c) sqrt(2 k_B T / m). Gaussian profile."""
    return nu0 / C * thermal_speed(T, m)


def doppler_width_wavelength(lam0: float, T: float, m: float) -> float:
    """Doppler width expressed in wavelength (m): delta_lam = (lam0/c) v_th."""
    return lam0 / C * thermal_speed(T, m)


def natural_width(A: float) -> float:
    """Natural (lifetime) Lorentzian half-width delta_nu = A / (4 pi) (Hz), from the
    spontaneous-decay rate A (s^-1)."""
    return A / (4.0 * math.pi)


def pressure_width(collision_rate: float) -> float:
    """Pressure (collisional) Lorentzian half-width delta_nu = nu_col / (2 pi) (Hz)."""
    return collision_rate / (2.0 * math.pi)


def dominant_mechanism(nu0: float, T: float, m: float, A: float,
                       collision_rate: float) -> str:
    """Name the widest of the three mechanisms: 'Doppler', 'natural', or 'pressure'."""
    widths = {
        "Doppler": doppler_width(nu0, T, m),
        "natural": natural_width(A),
        "pressure": pressure_width(collision_rate),
    }
    return max(widths, key=widths.get)


def lorentzian_fwhm(*half_widths: float) -> float:
    """Total Lorentzian FWHM (Hz): Lorentzian widths add linearly, and FWHM = 2 * the
    sum of the half-widths passed in."""
    return 2.0 * sum(half_widths)

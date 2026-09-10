"""Synchrotron radiation: the glow of relativistic electrons in magnetic fields.

Electrons spiralling in a magnetic field radiate. When they are relativistic the
emission beams forward into a narrow cone and shifts to high frequency --
synchrotron radiation, the source of most cosmic radio emission: jets, radio
galaxies, supernova remnants, pulsar wind nebulae.

Key quantities:

  * cyclotron (gyro) frequency:   nu_g = e B / (2 pi m_e)         (non-relativistic)
  * critical frequency:           nu_c ~ (3/2) gamma^2 nu_g sin(alpha)
    -- where a single electron's spectrum peaks; scales as gamma^2 B.
  * single-electron power:        P = (4/3) sigma_T c gamma^2 beta^2 U_B
    with U_B = B^2/(2 mu0) the magnetic energy density; scales as gamma^2 B^2.

A POWER-LAW electron energy distribution N(E) ~ E^{-p} radiates a power-law
spectrum S(nu) ~ nu^{-(p-1)/2}, so the observed spectral index alpha = (p-1)/2
reveals the electron population (p ~ 2.5 -> alpha ~ 0.75, typical of radio
sources). This module gives these frequencies, the power, and the spectral index.
SI units. Pure stdlib.
"""

from __future__ import annotations

import math

E_CHARGE = 1.602176634e-19
M_E = 9.1093837e-31
C = 2.99792458e8
MU0 = 4.0e-7 * math.pi
SIGMA_T = 6.6524587e-29


def gyrofrequency(B: float) -> float:
    """Non-relativistic electron cyclotron frequency nu_g = e B / (2 pi m_e)."""
    return E_CHARGE * B / (2.0 * math.pi * M_E)


def critical_frequency(gamma: float, B: float,
                       pitch_angle: float = math.pi / 2) -> float:
    """Synchrotron critical frequency nu_c = (3/2) gamma^2 nu_g sin(alpha),
    where the single-electron spectrum peaks."""
    return 1.5 * gamma * gamma * gyrofrequency(B) * math.sin(pitch_angle)


def single_electron_power(gamma: float, B: float) -> float:
    """Power radiated by one relativistic electron:
    P = (4/3) sigma_T c gamma^2 beta^2 U_B, U_B = B^2 / (2 mu0)."""
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    U_B = B * B / (2.0 * MU0)
    return (4.0 / 3.0) * SIGMA_T * C * gamma * gamma * beta2 * U_B


def cooling_time(gamma: float, B: float) -> float:
    """Synchrotron cooling time t = E / P = gamma m_e c^2 / P (s). Higher-energy
    electrons cool faster (t ~ 1/(gamma B^2))."""
    E = gamma * M_E * C * C
    return E / single_electron_power(gamma, B)


def spectral_index(p: float) -> float:
    """Observed synchrotron spectral index alpha = (p - 1)/2 for an electron
    energy distribution N(E) ~ E^{-p} (with S(nu) ~ nu^{-alpha})."""
    return (p - 1.0) / 2.0

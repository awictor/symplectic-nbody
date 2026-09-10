"""The de Broglie wavelength: matter as waves.

Louis de Broglie proposed that every particle has a wavelength inversely proportional
to its momentum,

    lambda = h / p,

so the more momentum a particle carries, the shorter its wave. This is why we do not
see the wave nature of everyday objects (a thrown baseball has lambda ~ 10^-34 m, far
below anything measurable) but do for electrons and atoms. Electron microscopes exploit
it directly: a 100 keV electron has lambda ~ 3.7 pm, thousands of times shorter than
visible light, so they resolve atoms.

For a non-relativistic particle of kinetic energy E, p = sqrt(2 m E), giving

    lambda = h / sqrt(2 m E),

and for a thermal gas the typical wavelength uses p ~ sqrt(3 m k_B T) (or the thermal de
Broglie wavelength h / sqrt(2 pi m k_B T)). The wave nature matters when lambda becomes
comparable to the interparticle spacing -- exactly the condition for quantum degeneracy
(Bose-Einstein condensation, electron degeneracy pressure).

This module gives the de Broglie wavelength from momentum, from kinetic energy, and for
a thermal particle, plus the momentum for a target wavelength, and reproduces the
electron-microscope and thermal-neutron wavelengths. SI units. Pure stdlib; the
matter-wave companion to the Compton and degeneracy modules.
"""

from __future__ import annotations

import math

H = 6.62607015e-34
K_B = 1.380649e-23
M_E = 9.1093837015e-31
M_P = 1.67262192e-27
M_N = 1.67492750e-27           # neutron mass
EV = 1.602176634e-19


def wavelength_from_momentum(p: float) -> float:
    """de Broglie wavelength lambda = h / p (m)."""
    return H / p


def wavelength_from_energy(E_kinetic: float, m: float) -> float:
    """Non-relativistic de Broglie wavelength from kinetic energy:
    lambda = h / sqrt(2 m E)."""
    return H / math.sqrt(2.0 * m * E_kinetic)


def wavelength_from_velocity(v: float, m: float) -> float:
    """de Broglie wavelength lambda = h / (m v) for a slow particle."""
    return H / (m * v)


def thermal_wavelength(T: float, m: float) -> float:
    """Thermal de Broglie wavelength lambda = h / sqrt(2 pi m k_B T) (m): the scale
    below which quantum statistics take over."""
    return H / math.sqrt(2.0 * math.pi * m * K_B * T)


def momentum_from_wavelength(lam: float) -> float:
    """Momentum p = h / lambda (kg m/s) for a target de Broglie wavelength."""
    return H / lam


def electron_microscope_wavelength(voltage_volts: float) -> float:
    """de Broglie wavelength (m) of an electron accelerated through a potential
    (non-relativistic): E = e V, lambda = h / sqrt(2 m_e e V)."""
    return wavelength_from_energy(voltage_volts * EV, M_E)

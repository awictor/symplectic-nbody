"""The photoelectric effect: light quantized into photons.

Shine light on a metal and it can eject electrons -- but only if the light's FREQUENCY
exceeds a threshold, no matter how intense a lower-frequency beam is. Classical waves
could not explain this; Einstein's 1905 resolution (his Nobel work) was that light
arrives in quanta of energy hf, and one photon gives all its energy to one electron:

    K_max = h f - phi,

where phi is the work function (the binding energy holding electrons in the metal) and
K_max the maximum kinetic energy of the ejected electron. Below the threshold frequency
f0 = phi/h nothing is emitted; above it, K_max rises linearly with frequency, with slope
exactly Planck's constant h -- independent of the metal. Brighter light ejects MORE
electrons but not FASTER ones.

The ejected electrons can be stopped by a retarding voltage; the stopping voltage
V_stop = K_max / e directly measures K_max, and Millikan's careful measurement of the
V_stop-vs-frequency line confirmed both h and the photon picture.

This module gives the photon energy, the maximum electron kinetic energy, the threshold
frequency and wavelength, and the stopping voltage, and reproduces the sodium/cesium
work-function thresholds and the linear stopping-voltage law. SI units, energies via an
eV helper. Pure stdlib; the quantum-of-light companion to the blackbody and Bohr modules.
"""

from __future__ import annotations

import math

H = 6.62607015e-34
C = 2.99792458e8
E_CHARGE = 1.602176634e-19
EV = 1.602176634e-19

# work functions (eV)
PHI_CESIUM = 2.14
PHI_SODIUM = 2.28
PHI_ZINC = 4.31
PHI_PLATINUM = 6.35


def photon_energy(freq: float) -> float:
    """Photon energy E = h f (joules)."""
    return H * freq


def photon_energy_from_wavelength(lam: float) -> float:
    """Photon energy E = h c / lambda (joules)."""
    return H * C / lam


def max_kinetic_energy(freq: float, work_function_ev: float) -> float:
    """Maximum ejected-electron kinetic energy K_max = h f - phi (joules). Negative
    (clamped to 0) below the threshold frequency -- no emission."""
    K = photon_energy(freq) - work_function_ev * EV
    return max(0.0, K)


def threshold_frequency(work_function_ev: float) -> float:
    """Threshold frequency f0 = phi / h (Hz): below this no electrons are ejected."""
    return work_function_ev * EV / H


def threshold_wavelength(work_function_ev: float) -> float:
    """Threshold wavelength lambda0 = h c / phi (m): the longest wavelength that can
    still eject electrons."""
    return H * C / (work_function_ev * EV)


def stopping_voltage(freq: float, work_function_ev: float) -> float:
    """Stopping voltage V_stop = K_max / e (volts): the retarding potential that just
    halts the fastest photoelectrons."""
    return max_kinetic_energy(freq, work_function_ev) / E_CHARGE


def is_emitting(freq: float, work_function_ev: float) -> bool:
    """True if the light frequency exceeds the metal's threshold (electrons ejected)."""
    return freq > threshold_frequency(work_function_ev)

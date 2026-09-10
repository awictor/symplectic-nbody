"""Debye shielding and the plasma frequency: what makes a plasma a plasma.

Drop a test charge into an ionized gas and the mobile electrons rearrange to screen
it: the potential no longer falls as 1/r but is cut off exponentially beyond the Debye
length

    lambda_D = sqrt( eps0 k_B T / (n e^2) ).

Inside lambda_D the charge is felt; outside, the plasma looks neutral. Two conditions
make a collection of charges behave as a plasma rather than as a gas of independent
particles: the system must be much larger than lambda_D (so screening operates), and
the number of particles inside a Debye sphere,

    N_D = (4/3) pi n lambda_D^3,

must be large (so the screening is statistical, collective, not dominated by a single
neighbour). N_D >> 1 is the plasma condition.

Disturb the electrons and they oscillate about the ions at the plasma (Langmuir)
frequency

    omega_p = sqrt( n e^2 / (eps0 m_e) ),

the natural ringing frequency of the electron sea. Electromagnetic waves below omega_p
cannot propagate and are reflected -- which is why the ionosphere bounces AM radio
around the curve of the Earth but lets higher-frequency FM and TV through, and why the
plasma frequency sets a cutoff throughout astrophysics.

This module gives the Debye length, the plasma parameter N_D, the plasma-condition
verdict, and the plasma frequency (angular and in Hz), and reproduces the ionospheric
radio cutoff and the many-particle Debye sphere. SI units. Pure stdlib; the collective-
plasma companion to the Saha and Alfven modules.
"""

from __future__ import annotations

import math

EPS0 = 8.8541878128e-12
K_B = 1.380649e-23
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31


def debye_length(n: float, T: float) -> float:
    """Debye screening length lambda_D = sqrt(eps0 k_B T / (n e^2)) (m). n is the
    electron number density (m^-3), T the electron temperature (K)."""
    return math.sqrt(EPS0 * K_B * T / (n * E_CHARGE ** 2))


def plasma_parameter(n: float, T: float) -> float:
    """Number of particles in a Debye sphere N_D = (4/3) pi n lambda_D^3. N_D >> 1
    is the condition for collective (plasma) behaviour."""
    lD = debye_length(n, T)
    return 4.0 / 3.0 * math.pi * n * lD ** 3


def is_plasma(n: float, T: float, size: float, N_min: float = 1.0) -> bool:
    """True if the ionized gas behaves as a plasma: the system is larger than the
    Debye length AND the Debye sphere is well-populated (N_D > N_min)."""
    return debye_length(n, T) < size and plasma_parameter(n, T) > N_min


def plasma_frequency(n: float) -> float:
    """Electron plasma (Langmuir) angular frequency omega_p = sqrt(n e^2 / eps0 m_e)
    (rad/s)."""
    return math.sqrt(n * E_CHARGE ** 2 / (EPS0 * M_E))


def plasma_frequency_hz(n: float) -> float:
    """Plasma frequency in Hz (omega_p / 2 pi). Waves below this are reflected."""
    return plasma_frequency(n) / (2.0 * math.pi)


def critical_density(freq_hz: float) -> float:
    """Electron density (m^-3) whose plasma frequency equals freq_hz: the density
    above which a wave of that frequency is reflected (the ionospheric cutoff)."""
    omega = 2.0 * math.pi * freq_hz
    return omega ** 2 * EPS0 * M_E / E_CHARGE ** 2

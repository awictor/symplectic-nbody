"""Alfven waves: the magnetized plasma's plucked string.

A magnetic field threading a conducting plasma acts like a set of tensioned
strings. Displace the field lines sideways and magnetic tension pulls them back,
while the plasma frozen to the lines supplies the inertia -- a transverse wave
that travels along B at the Alfven speed

    v_A = B / sqrt(mu0 rho),

with rho the mass density and mu0 the vacuum permeability. It is the magnetic
analogue of the speed of sound: replace the gas pressure that restores a sound
wave with magnetic tension B^2/mu0, and the mass density stays the same.

Two dimensionless numbers built from v_A govern magnetized flows. The plasma beta

    beta = p_gas / p_mag = n k_B T / (B^2 / 2 mu0)

compares thermal to magnetic pressure: beta << 1 (the solar corona) means the
field is in charge and channels the plasma; beta >> 1 (a stellar interior) means
gas pressure wins and drags the field around. The Alfven Mach number M_A = u/v_A
compares a bulk flow to the wave speed, setting where the solar wind goes
super-Alfvenic (past the Alfven surface, the corona can no longer magnetically
brake the wind).

Alfven waves carry energy and angular momentum out of the Sun, help heat the
corona to millions of kelvin, and are why the solar wind spins the Sun down.
This module gives v_A, the plasma beta, the Alfven Mach number, and the wave
travel time, and reproduces coronal and interplanetary values. SI units. Pure
stdlib; the magnetic-tension companion to the Larmor/synchrotron field physics.
"""

from __future__ import annotations

import math

MU0 = 1.25663706212e-6         # vacuum permeability (N/A^2)
K_B = 1.380649e-23
M_P = 1.6726219e-27            # proton mass (proton plasma: rho = n m_p)


def alfven_speed(B: float, rho: float) -> float:
    """Alfven speed v_A = B / sqrt(mu0 rho) (m/s), for mass density rho (kg/m^3)."""
    return B / math.sqrt(MU0 * rho)


def alfven_speed_number_density(B: float, n: float, m: float = M_P) -> float:
    """Alfven speed from a particle number density n (/m^3): rho = n m."""
    return alfven_speed(B, n * m)


def magnetic_pressure(B: float) -> float:
    """Magnetic pressure B^2 / (2 mu0) (Pa)."""
    return B * B / (2.0 * MU0)


def gas_pressure(n: float, T: float) -> float:
    """Ideal-gas pressure n k_B T (Pa) for number density n and temperature T."""
    return n * K_B * T


def plasma_beta(n: float, T: float, B: float) -> float:
    """Plasma beta = gas pressure / magnetic pressure. beta<1: field-dominated."""
    return gas_pressure(n, T) / magnetic_pressure(B)


def alfven_mach(u: float, B: float, rho: float) -> float:
    """Alfven Mach number M_A = u / v_A. >1 means super-Alfvenic flow."""
    return u / alfven_speed(B, rho)


def wave_travel_time(length: float, B: float, rho: float) -> float:
    """Time (s) for an Alfven wave to cross a distance `length` along the field."""
    return length / alfven_speed(B, rho)

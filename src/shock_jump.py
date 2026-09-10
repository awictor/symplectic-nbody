"""Sound speed and shock jumps: the Rankine-Hugoniot conditions.

Information in a gas travels at the sound speed c_s = sqrt(gamma P / rho) =
sqrt(gamma k_B T / (mu m_p)). When something moves faster than that -- a supernova
blast, a supersonic jet, a spacecraft on re-entry -- the gas cannot get out of the
way, and a shock forms: a razor-thin front across which density, pressure, velocity
and temperature jump discontinuously.

Conservation of mass, momentum and energy across the front (the Rankine-Hugoniot
conditions) fix every downstream quantity in terms of the upstream sonic Mach number
M1 = u1 / c_s1 and the adiabatic index gamma:

    density:      rho2/rho1 = (gamma+1) M1^2 / ((gamma-1) M1^2 + 2)
    pressure:     P2/P1     = (2 gamma M1^2 - (gamma-1)) / (gamma+1)
    temperature:  T2/T1     = (P2/P1)(rho1/rho2)

The compression saturates at (gamma+1)/(gamma-1) = 4 for a strong shock (gamma=5/3),
but the pressure and temperature jumps grow without bound as M1^2 -- which is why a
strong shock heats gas so ferociously (supernova remnants at 10^6-10^7 K, re-entry
plasma sheaths) even though it can only compress it fourfold. The downstream flow is
always subsonic in the shock frame (M2 < 1).

This module gives the sound speed and the density, pressure, temperature and
downstream-Mach jumps, and reproduces the strong-shock r=4 limit and the subsonic
post-shock flow. SI units. Pure stdlib; the shock-physics companion to the Sedov
blast-wave and Fermi-acceleration modules.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
M_P = 1.6726219e-27
GAMMA_MONO = 5.0 / 3.0         # monatomic / fully ionized gas
GAMMA_DIATOMIC = 7.0 / 5.0     # diatomic gas (air)


def sound_speed(T: float, mu: float = 0.6, gamma: float = GAMMA_MONO) -> float:
    """Adiabatic sound speed c_s = sqrt(gamma k_B T / (mu m_p)) (m/s)."""
    return math.sqrt(gamma * K_B * T / (mu * M_P))


def sound_speed_from_pressure(P: float, rho: float,
                              gamma: float = GAMMA_MONO) -> float:
    """Sound speed from pressure and density: c_s = sqrt(gamma P / rho) (m/s)."""
    return math.sqrt(gamma * P / rho)


def density_ratio(mach: float, gamma: float = GAMMA_MONO) -> float:
    """Rankine-Hugoniot density compression rho2/rho1. -> (gamma+1)/(gamma-1) at
    high Mach (=4 for gamma=5/3)."""
    m2 = mach * mach
    return (gamma + 1.0) * m2 / ((gamma - 1.0) * m2 + 2.0)


def pressure_ratio(mach: float, gamma: float = GAMMA_MONO) -> float:
    """Rankine-Hugoniot pressure jump P2/P1 = (2 gamma M^2 - (gamma-1))/(gamma+1).
    Grows without bound as M^2."""
    return (2.0 * gamma * mach * mach - (gamma - 1.0)) / (gamma + 1.0)


def temperature_ratio(mach: float, gamma: float = GAMMA_MONO) -> float:
    """Temperature jump T2/T1 = (P2/P1)(rho1/rho2) across the shock."""
    return pressure_ratio(mach, gamma) / density_ratio(mach, gamma)


def downstream_mach(mach: float, gamma: float = GAMMA_MONO) -> float:
    """Downstream Mach number M2 in the shock frame:
    M2^2 = ((gamma-1) M1^2 + 2) / (2 gamma M1^2 - (gamma-1)). Always < 1 for M1 > 1."""
    m2 = mach * mach
    m2_2 = ((gamma - 1.0) * m2 + 2.0) / (2.0 * gamma * m2 - (gamma - 1.0))
    return math.sqrt(m2_2)


def strong_shock_compression(gamma: float = GAMMA_MONO) -> float:
    """Limiting density compression (gamma+1)/(gamma-1) as M -> infinity (=4)."""
    return (gamma + 1.0) / (gamma - 1.0)

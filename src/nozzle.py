"""The de Laval nozzle: how a converging-diverging throat makes gas go supersonic.

To push exhaust out faster than sound -- the whole point of a rocket or a supersonic wind
tunnel -- you cannot just keep narrowing a pipe. Subsonic flow speeds up as the area shrinks,
but once it hits Mach 1 the behaviour flips: supersonic flow speeds up as the area *grows*.
So the gas must be squeezed to sonic conditions at a minimum-area throat and then expanded
through a widening bell. That is the de Laval nozzle, and it is governed by the isentropic
relations for a calorically perfect gas.

Along the nozzle the stagnation (chamber) conditions are fixed, and everything follows from
the local Mach number M and the ratio of specific heats gamma. Writing
b = 1 + (gamma-1)/2 M^2,

    T0/T = b,      P0/P = b^(gamma/(gamma-1)),      rho0/rho = b^(1/(gamma-1)),

and the area needed to reach a given Mach number, relative to the sonic throat area A*, is
the area-Mach relation

    A/A* = (1/M) [ (2/(gamma+1)) b ]^((gamma+1)/(2(gamma-1))).

The throat chokes at M = 1: once the pressure ratio is large enough, the mass flow saturates
at a maximum set by the throat area and chamber conditions and no longer rises as the exit
pressure falls. The exit Mach number is then a pure function of the area ratio, and the whole
device converts chamber enthalpy into directed kinetic energy at the exhaust.

This module gives the isentropic temperature/pressure/density ratios, the area-Mach relation
and its supersonic inversion, the choked mass flow, the exhaust velocity, and the critical
(choking) pressure ratio, and reproduces the ~0.528 choking ratio for air and the way a big
area ratio buys a high exit Mach number. SI units. Pure stdlib; the compressible-flow
companion to the Mach-cone and shock-jump notes.
"""

from __future__ import annotations

import math

GAMMA_AIR = 1.4
R_AIR = 287.0                 # specific gas constant of air (J/(kg K))


def temperature_ratio(mach: float, gamma: float = GAMMA_AIR) -> float:
    """Stagnation-to-static temperature ratio T0/T = 1 + (gamma-1)/2 M^2."""
    return 1.0 + 0.5 * (gamma - 1.0) * mach * mach


def pressure_ratio(mach: float, gamma: float = GAMMA_AIR) -> float:
    """Stagnation-to-static pressure ratio P0/P = (T0/T)^(gamma/(gamma-1))."""
    return temperature_ratio(mach, gamma) ** (gamma / (gamma - 1.0))


def density_ratio(mach: float, gamma: float = GAMMA_AIR) -> float:
    """Stagnation-to-static density ratio rho0/rho = (T0/T)^(1/(gamma-1))."""
    return temperature_ratio(mach, gamma) ** (1.0 / (gamma - 1.0))


def area_ratio(mach: float, gamma: float = GAMMA_AIR) -> float:
    """Area-Mach relation A/A*: the local area relative to the sonic throat needed to reach
    Mach M. Minimum (=1) at M=1; rises on both the subsonic and supersonic sides."""
    b = temperature_ratio(mach, gamma)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (1.0 / mach) * (2.0 / (gamma + 1.0) * b) ** exponent


def critical_pressure_ratio(gamma: float = GAMMA_AIR) -> float:
    """Critical (choking) static-to-stagnation pressure ratio P*/P0 = (2/(gamma+1))^
    (gamma/(gamma-1)). ~0.528 for air: below this back-pressure ratio the throat chokes."""
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def mach_from_area_ratio(area_ratio_target: float, supersonic: bool = True,
                         gamma: float = GAMMA_AIR) -> float:
    """Invert the area-Mach relation for the Mach number at a given A/A*, choosing the
    supersonic branch (M>1, the diverging section) or subsonic branch (M<1). Bisection."""
    if area_ratio_target < 1.0:
        raise ValueError("A/A* must be >= 1")
    if supersonic:
        lo, hi = 1.0000001, 100.0
    else:
        lo, hi = 1e-6, 0.9999999

    def f(m):
        return area_ratio(m, gamma) - area_ratio_target

    # both branches are monotonic in the chosen interval
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        # subsonic: A/A* decreases with M; supersonic: increases with M
        if supersonic:
            if fm < 0.0:
                lo = mid
            else:
                hi = mid
        else:
            if fm < 0.0:
                hi = mid
            else:
                lo = mid
    return 0.5 * (lo + hi)


def choked_mass_flow(p0: float, t0: float, throat_area: float,
                     gamma: float = GAMMA_AIR, r: float = R_AIR) -> float:
    """Choked (maximum) mass flow rate (kg/s) through a throat of area A* with chamber
    stagnation pressure P0 and temperature T0:
    mdot = A* P0 sqrt(gamma/(R T0)) (2/(gamma+1))^((gamma+1)/(2(gamma-1)))."""
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return (throat_area * p0 * math.sqrt(gamma / (r * t0))
            * (2.0 / (gamma + 1.0)) ** exponent)


def exhaust_velocity(t0: float, mach_exit: float, gamma: float = GAMMA_AIR,
                     r: float = R_AIR) -> float:
    """Exit velocity (m/s) = M_e sqrt(gamma R T_e), with the exit static temperature
    T_e = T0 / (1 + (gamma-1)/2 M_e^2) from isentropic expansion of a chamber at T0."""
    t_exit = t0 / temperature_ratio(mach_exit, gamma)
    return mach_exit * math.sqrt(gamma * r * t_exit)

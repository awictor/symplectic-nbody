"""Magnetic braking and gyrochronology: reading a star's age from its spin.

A magnetized wind is a superb brake. Plasma leaving the star stays locked to the
field lines out to the Alfven radius, so it is forced to corotate with the star far
beyond the surface -- and carries off angular momentum with a lever arm r_A that is
many stellar radii long. A tiny mass loss thus bleeds a large spin. Because the
torque grows steeply with rotation rate, fast rotators brake hard and slow ones
barely at all, so an initially wide spread of stellar spins converges onto a single
sequence.

The empirical result (Skumanich 1972) is that a Sun-like star's surface rotation
decays as

    Omega(t) ~ t^(-1/2),   equivalently   v_rot ~ t^(-1/2),

so the rotation *period* grows as P ~ t^(1/2). Run backwards, a measured period
dates the star:

    t = t_sun (P / P_sun)^2,

which is gyrochronology -- the clock that ages field stars and open clusters from a
single rotation measurement. The physics behind the exponent is the Weber-Davis
angular-momentum loss dJ/dt = (2/3) Mdot Omega r_A^2 with a dynamo field B ~ Omega
setting r_A; feeding that into J = I Omega gives the power law.

This module gives the Skumanich spin-down, the gyrochronology age, the Weber-Davis
wind torque and the associated spin-down time, and reproduces the Sun's ~25-day
period at 4.6 Gyr. SI units. Pure stdlib; the angular-momentum sink of the
Alfvenic, Parker-spiralled solar wind.
"""

from __future__ import annotations

import math

DAY = 86400.0
YEAR = 3.15576e7
GYR = 1e9 * YEAR

# Solar calibration anchors for gyrochronology.
P_SUN = 25.4 * DAY            # solar (sidereal, equatorial) rotation period
T_SUN = 4.567 * GYR           # solar age
OMEGA_SUN = 2.0 * math.pi / P_SUN


def rotation_rate(t: float, omega0: float = OMEGA_SUN, t0: float = T_SUN) -> float:
    """Skumanich spin-down Omega(t) = omega0 sqrt(t0/t): rotation fades as t^(-1/2)."""
    return omega0 * math.sqrt(t0 / t)


def rotation_period(t: float, p0: float = P_SUN, t0: float = T_SUN) -> float:
    """Rotation period grows as P(t) = p0 sqrt(t/t0)."""
    return p0 * math.sqrt(t / t0)


def gyro_age(period: float, p_sun: float = P_SUN, t_sun: float = T_SUN) -> float:
    """Gyrochronology: age (s) from a measured rotation period,
    t = t_sun (P/P_sun)^2. Invert the Skumanich law."""
    return t_sun * (period / p_sun) ** 2


def wind_torque(mdot: float, omega: float, r_alfven: float) -> float:
    """Weber-Davis magnetized-wind torque magnitude |dJ/dt| = (2/3) Mdot Omega r_A^2.
    The Alfven radius r_A is the lever arm over which the wind corotates."""
    return (2.0 / 3.0) * mdot * omega * r_alfven ** 2


def spin_down_time(I: float, omega: float, mdot: float, r_alfven: float) -> float:
    """Order-of-magnitude spin-down time J / |dJ/dt| = I Omega / torque (s)."""
    return I * omega / wind_torque(mdot, omega, r_alfven)


def alfven_radius(B_surf: float, R_star: float, mdot: float, v_wind: float) -> float:
    """Alfven radius r_A (m) for a split-monopole wind, where the flow speed equals
    the local Alfven speed. With a radial field B_r = B_surf (R/r)^2 and mass flux
    Mdot = 4 pi r^2 rho v, setting v^2 = B_r^2/(mu0 rho) gives

        r_A = sqrt(4 pi B_surf^2 R_star^4 / (mu0 Mdot v_wind)).

    For solar values (B ~ 1-2 G, Mdot ~ 2e9 kg/s, v ~ 400 km/s) this is ~10 R_sun,
    the long lever arm that makes the magnetized wind such an efficient brake."""
    MU0 = 1.25663706212e-6
    return math.sqrt(4.0 * math.pi * B_surf ** 2 * R_star ** 4
                     / (MU0 * mdot * v_wind))

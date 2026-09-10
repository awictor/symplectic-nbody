"""The Hohmann transfer: the cheapest two-burn orbit change.

To move between two circular orbits, the most fuel-efficient maneuver (for a
moderate radius ratio) is the Hohmann transfer: a burn onto an ellipse whose
periapsis touches the inner orbit and apoapsis the outer, then a second burn to
circularize. The two velocity changes are

    dv1 = sqrt(mu/r1) ( sqrt( 2 r2/(r1+r2) ) - 1 )      (raise onto the ellipse)
    dv2 = sqrt(mu/r2) ( 1 - sqrt( 2 r1/(r1+r2) ) )      (circularize at r2)

and the trip takes half the transfer ellipse's period,

    t = pi sqrt( a_t^3 / mu ),   a_t = (r1 + r2)/2.

This is the backbone of mission design: it sets the delta-v budget (hence the
propellant, via the rocket equation) for going from low Earth orbit to
geostationary, or from Earth's orbit to Mars's. This module computes the burns,
the transfer time, and the launch window phase angle, and reproduces the standard
LEO->GEO (~3.9 km/s) and Earth->Mars (~5.6 km/s, ~259 day) numbers. SI units.
Pure stdlib.
"""

from __future__ import annotations

import math

MU_EARTH = 3.986004418e14      # m^3/s^2
MU_SUN = 1.32712440018e20      # m^3/s^2
AU = 1.495978707e11            # m
DAY = 86400.0


def transfer(r1: float, r2: float, mu: float):
    """Return (dv1, dv2, dv_total, transfer_time) for a Hohmann transfer from a
    circular orbit of radius r1 to one of radius r2 about a body of parameter mu.
    Radii in m, speeds in m/s, time in s."""
    a_t = 0.5 * (r1 + r2)
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)
    dv1 = v1 * (math.sqrt(2.0 * r2 / (r1 + r2)) - 1.0)
    dv2 = v2 * (1.0 - math.sqrt(2.0 * r1 / (r1 + r2)))
    dv_total = abs(dv1) + abs(dv2)
    t_transfer = math.pi * math.sqrt(a_t ** 3 / mu)
    return dv1, dv2, dv_total, t_transfer


def phase_angle(r1: float, r2: float, mu: float) -> float:
    """Required lead angle (radians) of the target ahead of the chaser at the
    first burn, so the target arrives at apoapsis when the spacecraft does.
    alpha = pi - omega_target * t_transfer."""
    _dv1, _dv2, _dvt, t = transfer(r1, r2, mu)
    omega2 = math.sqrt(mu / r2 ** 3)     # target angular velocity
    return math.pi - omega2 * t


def rocket_equation_mass_ratio(dv: float, v_exhaust: float) -> float:
    """Tsiolkovsky: initial/final mass ratio m0/mf = exp(dv / v_exhaust) needed to
    supply a total delta-v with a given exhaust velocity."""
    return math.exp(dv / v_exhaust)

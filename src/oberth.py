"""The Oberth effect: why rockets burn deep in a gravity well.

A rocket burn changes speed by dv, but the change in KINETIC energy per unit mass
depends on how fast you are already going:

    dE = 1/2 (v + dv)^2 - 1/2 v^2 = v dv + 1/2 dv^2.

The v dv term means the SAME dv buys MORE energy when the ship is moving fast --
which it is at periapsis, deep in the gravity well. Burning there (the "Oberth
maneuver") is dramatically more efficient than burning far out, and it is why
interplanetary missions dive toward a planet before their escape burn and why
powered gravity assists work.

This module computes the energy gain of a burn, the resulting hyperbolic excess
speed (v_infinity) after an escape burn, and shows the periapsis-vs-apoapsis
advantage. SI units. Pure stdlib.
"""

from __future__ import annotations

import math

MU_EARTH = 3.986004418e14      # m^3/s^2
R_EARTH = 6.371e6


def energy_gain(v: float, dv: float) -> float:
    """Specific kinetic-energy gain from a burn dv at current speed v:
    dE = v dv + 1/2 dv^2."""
    return v * dv + 0.5 * dv * dv


def speed_at_radius(mu: float, r: float, a: float) -> float:
    """Vis-viva speed on an orbit of semi-major axis a at radius r
    (a -> inf for a parabolic/escape orbit)."""
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def v_infinity_after_burn(mu: float, r_burn: float, v_before: float,
                          dv: float) -> float:
    """Hyperbolic excess speed (speed at infinity) after adding dv at radius
    r_burn. Specific orbital energy eps = v^2/2 - mu/r; v_inf = sqrt(2 eps) if
    eps > 0 (escaping), else 0 (still bound)."""
    v_after = v_before + dv
    eps = 0.5 * v_after * v_after - mu / r_burn
    return math.sqrt(2.0 * eps) if eps > 0.0 else 0.0


def oberth_advantage(mu: float, r_peri: float, r_apo: float, dv: float,
                     a: float) -> float:
    """Ratio of energy gained by burning dv at periapsis vs at apoapsis on an
    orbit of semi-major axis a. > 1 because periapsis speed is higher."""
    v_peri = speed_at_radius(mu, r_peri, a)
    v_apo = speed_at_radius(mu, r_apo, a)
    return energy_gain(v_peri, dv) / energy_gain(v_apo, dv)

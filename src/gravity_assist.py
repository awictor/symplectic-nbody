"""Gravity assist: stealing orbital energy from a planet.

A spacecraft flying past a planet follows a hyperbola in the planet's frame: it comes
in and leaves at the same speed v_inf relative to the planet (energy is conserved
there), but its direction is bent by the turn angle delta. In the SUN's frame, though,
the planet is moving, so rotating the velocity vector while the planet drags it along
changes the craft's heliocentric speed -- a free boost (or brake) worth up to twice the
planet's orbital speed.

The hyperbolic turn angle depends on how close and how fast the approach is:

    sin(delta/2) = 1 / e,      e = 1 + r_p v_inf^2 / mu,

with r_p the periapsis distance, v_inf the hyperbolic excess speed, and mu the planet's
gravitational parameter. A slow, deep pass bends the trajectory most. The maximum
heliocentric speed change is

    dv_max = 2 v_inf sin(delta/2),

approached for a head-on geometry; a full reversal (delta = 180 deg) would add 2 v_inf.
Voyager 2 used exactly this to climb from ~10 km/s to Solar-System-escape speed by
chaining Jupiter, Saturn, Uranus and Neptune, and every deep-space mission since leans
on it. The planet pays for the boost with an utterly negligible loss of orbital energy.

This module gives the eccentricity, the turn angle, the maximum speed change, and the
new heliocentric speed after a flyby, and reproduces the Voyager-class boost. SI units.
Pure stdlib; the flyby companion to the Hohmann and Oberth modules.
"""

from __future__ import annotations

import math

MU_JUPITER = 1.26686534e17     # Jupiter's gravitational parameter (m^3/s^2)
MU_EARTH = 3.986004418e14
V_JUPITER = 13.07e3            # Jupiter's orbital speed (m/s)


def eccentricity(v_inf: float, r_p: float, mu: float) -> float:
    """Hyperbolic eccentricity e = 1 + r_p v_inf^2 / mu (>1). A slower or deeper pass
    gives e closer to 1 and a sharper bend."""
    return 1.0 + r_p * v_inf * v_inf / mu


def turn_angle(v_inf: float, r_p: float, mu: float) -> float:
    """Deflection (turn) angle delta (radians): sin(delta/2) = 1/e. The angle the
    velocity vector swings through during the flyby."""
    e = eccentricity(v_inf, r_p, mu)
    return 2.0 * math.asin(1.0 / e)


def max_delta_v(v_inf: float, r_p: float, mu: float) -> float:
    """Maximum heliocentric speed change from a single flyby:
    dv = 2 v_inf sin(delta/2) = 2 v_inf / e. Head-on geometry, best case."""
    e = eccentricity(v_inf, r_p, mu)
    return 2.0 * v_inf / e


def heliocentric_speed_after(v_sc: float, v_planet: float, v_inf: float,
                             r_p: float, mu: float,
                             approach_angle: float = math.pi) -> float:
    """Heliocentric speed (m/s) after a flyby, adding the rotated v_inf vector to the
    planet's velocity. approach_angle=pi is the optimal head-on trailing pass. Uses the
    law of cosines with the turn angle delta."""
    delta = turn_angle(v_inf, r_p, mu)
    # incoming v_inf makes angle `approach_angle` with the planet's motion; after the
    # turn it is rotated by delta. Best-case gain adds up to 2 v_inf sin(delta/2).
    gain = 2.0 * v_inf * math.sin(delta / 2.0) * math.sin(approach_angle / 2.0)
    return v_sc + gain


def slingshot_gain(v_inf: float, r_p: float, mu: float) -> float:
    """The heliocentric speed gain (m/s) of an optimal trailing flyby = max_delta_v."""
    return max_delta_v(v_inf, r_p, mu)

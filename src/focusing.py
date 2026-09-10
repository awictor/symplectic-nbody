"""Gravitational focusing: why planets grow faster than geometry allows.

Two bodies approaching at relative speed v_inf do not have to hit head-on to
collide -- gravity bends their trajectories inward. The effective collision
cross-section is enhanced over the geometric pi R^2 by the GRAVITATIONAL
FOCUSING factor,

    sigma = pi R^2 ( 1 + v_esc^2 / v_inf^2 ),

where v_esc is the surface escape speed of the combined body. The dimensionless
ratio

    Theta = v_esc^2 / (2 v_inf^2)     (the Safronov number)

measures how much focusing helps. When Theta >> 1 (slow encounters, large bodies)
the cross-section far exceeds the geometric one, so big bodies sweep up mass ever
faster -- RUNAWAY GROWTH, the mechanism that builds planetary embryos out of a
planetesimal swarm. At high v_inf gravity is negligible and sigma -> pi R^2.

This module gives the focused cross-section, the enhancement factor, the Safronov
number, and reproduces the geometric and runaway limits. SI units. Pure stdlib;
complements cosmic_velocities.
"""

from __future__ import annotations

import math

G = 6.67430e-11


def escape_speed(M: float, R: float) -> float:
    """Surface escape speed sqrt(2 G M / R)."""
    return math.sqrt(2.0 * G * M / R)


def geometric_cross_section(R: float) -> float:
    """Geometric cross-section pi R^2."""
    return math.pi * R * R


def focusing_factor(M: float, R: float, v_inf: float) -> float:
    """Gravitational-focusing enhancement 1 + v_esc^2 / v_inf^2."""
    ve = escape_speed(M, R)
    return 1.0 + (ve * ve) / (v_inf * v_inf)


def collision_cross_section(M: float, R: float, v_inf: float) -> float:
    """Effective cross-section sigma = pi R^2 (1 + v_esc^2/v_inf^2)."""
    return geometric_cross_section(R) * focusing_factor(M, R, v_inf)


def safronov_number(M: float, R: float, v_inf: float) -> float:
    """Safronov number Theta = v_esc^2 / (2 v_inf^2). Theta >> 1 -> runaway
    growth; Theta << 1 -> geometric (ordered) growth."""
    ve = escape_speed(M, R)
    return (ve * ve) / (2.0 * v_inf * v_inf)


def is_runaway(M: float, R: float, v_inf: float) -> bool:
    """Runaway-growth regime when the Safronov number exceeds 1 (focusing
    dominates the geometric cross-section)."""
    return safronov_number(M, R, v_inf) > 1.0

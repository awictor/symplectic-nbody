"""Frame-dragging and geodetic precession: the Gravity Probe B effects.

A gyroscope orbiting a massive body precesses for two general-relativistic
reasons, both measured by Gravity Probe B (2011):

GEODETIC (de Sitter) precession -- from the curvature of space the gyro is
carried through as it orbits:

    Omega_geo = (3/2) (G M / (c^2 r)) * v_orb / r     (per orbit-averaged),
              = (3/2) sqrt( (G M)^3 / (c^4 r^5) ).

FRAME-DRAGGING (Lense-Thirring) precession -- from the rotating body dragging
spacetime around with it:

    Omega_LT = G J / (c^2 r^3) * (factor of order 1).

For a gyroscope in a polar orbit at Gravity Probe B's altitude the predictions
are ~6600 milliarcsec/yr (geodetic) and ~39 mas/yr (frame-dragging), both
confirmed. The frame-dragging term is far smaller, which is why it took a
dedicated experiment to see. This module computes both rates and reproduces the
Gravity Probe B numbers. SI units. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
M_EARTH = 5.972e24
R_EARTH = 6.371e6
J_EARTH = 5.86e33              # Earth's spin angular momentum, kg m^2 / s

# radians -> milliarcseconds per (given) time
RAD_TO_MAS = (180.0 / math.pi) * 3600.0 * 1000.0
YEAR = 3.15576e7


def geodetic_rate(M: float, r: float) -> float:
    """Geodetic (de Sitter) precession rate (rad/s) for a circular orbit:
    Omega = (3/2) sqrt((G M)^3 / (c^4 r^5))."""
    return 1.5 * math.sqrt((G * M) ** 3 / (C ** 4 * r ** 5))


def frame_dragging_rate(J: float, r: float) -> float:
    """Lense-Thirring frame-dragging precession rate (rad/s), orbit-averaged for a
    POLAR orbit. The vector rate is (G/c^2 r^3)[3(J.rhat)rhat - J]; averaging the
    perpendicular component over a polar orbit gives the factor 1/2:
        Omega = (1/2) G J / (c^2 r^3)."""
    return 0.5 * G * J / (C ** 2 * r ** 3)


def geodetic_mas_per_year(M: float, r: float) -> float:
    """Geodetic precession in milliarcseconds per year."""
    return geodetic_rate(M, r) * YEAR * RAD_TO_MAS


def frame_dragging_mas_per_year(J: float, r: float) -> float:
    """Frame-dragging precession in milliarcseconds per year."""
    return frame_dragging_rate(J, r) * YEAR * RAD_TO_MAS


def gravity_probe_b_altitude() -> float:
    """Gravity Probe B orbital radius (642 km altitude polar orbit), in meters."""
    return R_EARTH + 642e3

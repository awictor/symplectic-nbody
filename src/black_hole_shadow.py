"""The black-hole shadow: the dark disk the Event Horizon Telescope imaged.

A black hole casts a shadow larger than its event horizon. Light passing too close is
captured; the boundary of capture, seen from far away, is set not by the horizon but by
the unstable photon orbit. For a non-spinning (Schwarzschild) hole the photon sphere
sits at

    r_photon = 3 G M / c^2 = 1.5 r_s,

and gravitational lensing magnifies the apparent shadow to a critical impact parameter

    b_crit = 3 sqrt(3) G M / c^2 = 3 sqrt(3) / 2 * r_s,

so the shadow's angular DIAMETER on the sky is

    theta = 2 b_crit / D = 6 sqrt(3) G M / (c^2 D),

with D the distance to the hole. That is about 5.2 Schwarzschild radii across -- larger
than the ~2 r_s horizon because of the light-bending. Plugging in M87* (6.5e9 solar
masses at 16.8 Mpc) gives ~40 microarcseconds, and Sgr A* (4.15e6 solar masses at 8.15
kpc) ~52 microarcseconds -- exactly the sizes the EHT measured, resolving structure on
the scale of a few tens of microarcseconds by linking radio dishes across the Earth.

This module gives the photon-sphere and critical-impact-parameter radii, the shadow's
physical and angular size, and the M87*/Sgr A* predictions, and reproduces the ~40-50
microarcsecond EHT measurements. SI units, angles convertible to microarcseconds. Pure
stdlib; the strong-lensing companion to the Schwarzschild and lensing modules.
"""

from __future__ import annotations

import math

G = 6.67430e-11
C = 2.99792458e8
M_SUN = 1.989e30
PC = 3.0856775814913673e16
MPC = 1e6 * PC

# radians -> microarcseconds
UAS_PER_RAD = 180.0 / math.pi * 3600.0 * 1e6


def schwarzschild_radius(M: float) -> float:
    """Schwarzschild radius r_s = 2 G M / c^2 (m)."""
    return 2.0 * G * M / (C * C)


def photon_sphere_radius(M: float) -> float:
    """Photon-sphere radius r = 3 G M / c^2 = 1.5 r_s (m): the unstable circular
    light orbit."""
    return 3.0 * G * M / (C * C)


def critical_impact_parameter(M: float) -> float:
    """Critical impact parameter b_crit = 3 sqrt(3) G M / c^2 (m): the apparent radius
    of the shadow (light inside this is captured)."""
    return 3.0 * math.sqrt(3.0) * G * M / (C * C)


def shadow_diameter(M: float) -> float:
    """Physical diameter of the shadow, 2 b_crit = 6 sqrt(3) G M / c^2 (m)."""
    return 2.0 * critical_impact_parameter(M)


def shadow_angular_diameter(M: float, D: float) -> float:
    """Angular diameter of the shadow (radians) at distance D: theta = 2 b_crit / D."""
    return shadow_diameter(M) / D


def shadow_angular_diameter_uas(M: float, D: float) -> float:
    """Shadow angular diameter in microarcseconds."""
    return shadow_angular_diameter(M, D) * UAS_PER_RAD


def shadow_in_rs(M: float) -> float:
    """Shadow diameter in units of the Schwarzschild radius (= 3 sqrt(3) ~ 5.196)."""
    return shadow_diameter(M) / schwarzschild_radius(M)

"""The Mach cone: the geometry of going supersonic.

When a source moves faster than sound, the pressure disturbances it emits can no longer race
ahead of it. Each spherical wavelet, expanding at the sound speed c while the source runs on
at speed U, is left behind, and their common envelope is a cone trailing the source -- the
Mach cone. Its half-angle is fixed by the Mach number M = U/c through a beautifully simple
relation,

    sin(mu) = c / U = 1 / M,

so the faster you go the tighter the cone: 90 degrees (a flat disturbance front) right at
M = 1, closing to 30 degrees at M = 2 and 11.5 degrees at M = 5. The cone is the shock front
whose passage a ground observer hears as a sonic boom -- and because it trails at a fixed
angle, an aircraft at altitude H lays the boom down on the ground a distance H/tan(mu) *behind*
its overhead point, arriving a time H/(U tan(mu)) after it has already passed.

Two more compressibility results ride along. Below Mach 1, thin-airfoil lift and pressure
grow by the Prandtl-Glauert factor 1/sqrt(1 - M^2) as the flow stiffens toward the sound
barrier. Above Mach 1, a supersonic flow turning around a convex corner accelerates through
a Prandtl-Meyer expansion fan, its turn angle given by the Prandtl-Meyer function nu(M).

This module gives the Mach angle, the Mach number from the cone, the sonic-boom lag and
ground offset, the Prandtl-Glauert factor, and the Prandtl-Meyer angle, and reproduces the
90-degrees-at-M1 to tightening-cone behaviour and the subsonic lift divergence. SI units,
angles in radians unless _deg. Pure stdlib; the supersonic-geometry companion to the
shock-jump and sound-speed notes.
"""

from __future__ import annotations

import math

GAMMA_AIR = 1.4


def mach_angle(mach: float) -> float:
    """Mach-cone half-angle mu = arcsin(1/M) (rad), defined only for M >= 1. Equals pi/2 at
    M = 1 (the disturbance front is flat) and shrinks toward 0 as M grows."""
    if mach < 1.0:
        raise ValueError("Mach cone exists only for M >= 1")
    return math.asin(1.0 / mach)


def mach_angle_deg(mach: float) -> float:
    """Mach angle in degrees: 90 at M=1, 30 at M=2, ~11.5 at M=5."""
    return math.degrees(mach_angle(mach))


def mach_from_angle(mu_rad: float) -> float:
    """Recover the Mach number from a measured cone half-angle: M = 1 / sin(mu)."""
    return 1.0 / math.sin(mu_rad)


def sonic_boom_ground_offset(altitude: float, mach: float) -> float:
    """Horizontal distance (m) behind the overhead point at which the Mach cone reaches the
    ground from an aircraft at the given altitude: x = altitude / tan(mu)."""
    return altitude / math.tan(mach_angle(mach))


def sonic_boom_delay(altitude: float, mach: float, sound_speed: float) -> float:
    """Time (s) after an aircraft passes overhead before a ground observer hears the boom:
    t = altitude / (U tan(mu)), with U = M * sound_speed."""
    u = mach * sound_speed
    return altitude / (u * math.tan(mach_angle(mach)))


def prandtl_glauert_factor(mach: float) -> float:
    """Subsonic compressibility correction 1 / sqrt(1 - M^2): the factor by which thin-
    airfoil lift and pressure coefficients grow over their incompressible values as M -> 1.
    Defined for M < 1."""
    if mach >= 1.0:
        raise ValueError("Prandtl-Glauert applies only for M < 1")
    return 1.0 / math.sqrt(1.0 - mach * mach)


def prandtl_meyer_angle(mach: float, gamma: float = GAMMA_AIR) -> float:
    """Prandtl-Meyer function nu(M) (rad): the angle through which a sonic (M=1) supersonic
    flow must expand to reach Mach M around a convex corner. Zero at M=1, rising toward a
    finite maximum as M -> infinity."""
    if mach < 1.0:
        raise ValueError("Prandtl-Meyer function defined only for M >= 1")
    gp = (gamma + 1.0) / (gamma - 1.0)
    m2 = mach * mach - 1.0
    return math.sqrt(gp) * math.atan(math.sqrt(m2 / gp)) - math.atan(math.sqrt(m2))

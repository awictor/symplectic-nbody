"""Snell's law: how light bends, traps, and reflects at a boundary.

Light changes speed when it crosses between media, and to keep its wavefronts continuous it
must change direction. Snell's law states the exact bend:

    n1 sin(theta1) = n2 sin(theta2),

where n is the refractive index (c / v in the medium) and theta the angle from the normal.
Going into a denser medium (n2 > n1) light bends toward the normal; coming out it bends away.

Push the outgoing case far enough and something dramatic happens. Above the critical angle

    theta_c = arcsin(n2 / n1)        (n1 > n2)

the refracted ray would need sin(theta2) > 1, which is impossible, so all the light is
reflected back inside -- total internal reflection. That perfect mirror is what guides light
down an optical fibre for thousands of kilometres and makes a diamond (n = 2.42, theta_c ~
24 deg) sparkle by trapping and re-releasing light.

At the boundary the reflected and refracted beams also carry different polarizations, and at
Brewster's angle theta_B = arctan(n2 / n1) the reflected light is perfectly polarized -- the
principle behind polarizing sunglasses and camera filters that kill glare.

This module gives the refraction angle, the critical angle and a total-internal-reflection
test, Brewster's angle, the refractive index from the speed of light, and the fibre
acceptance angle (numerical aperture), and reproduces water's ~48.6 deg critical angle and a
diamond's sparkle. SI units, angles in radians unless _deg. Pure stdlib; the ray-optics
companion to the diffraction-limit and Bragg notes.
"""

from __future__ import annotations

import math

C = 299792458.0               # speed of light in vacuum (m/s)
N_VACUUM = 1.0
N_AIR = 1.000293
N_WATER = 1.333
N_GLASS = 1.52
N_DIAMOND = 2.417


def refraction_angle(n1: float, theta1: float, n2: float) -> float:
    """Refracted angle theta2 (rad) from Snell's law n1 sin(theta1) = n2 sin(theta2):
    theta2 = arcsin(n1 sin(theta1) / n2). Raises on total internal reflection (argument > 1)."""
    s = n1 * math.sin(theta1) / n2
    if s > 1.0:
        raise ValueError("total internal reflection: no refracted ray")
    return math.asin(s)


def critical_angle(n1: float, n2: float) -> float:
    """Critical angle theta_c = arcsin(n2 / n1) (rad) for total internal reflection, when
    going from denser n1 into rarer n2 (requires n1 > n2)."""
    if n1 <= n2:
        raise ValueError("no critical angle unless n1 > n2")
    return math.asin(n2 / n1)


def is_total_internal_reflection(n1: float, theta1: float, n2: float) -> bool:
    """True if a ray from medium n1 at incidence theta1 undergoes total internal reflection
    at the boundary with rarer medium n2 (n1 > n2 and theta1 >= theta_c)."""
    if n1 <= n2:
        return False
    return theta1 >= critical_angle(n1, n2)


def brewster_angle(n1: float, n2: float) -> float:
    """Brewster's angle theta_B = arctan(n2 / n1) (rad): the incidence angle at which the
    reflected light is perfectly polarized."""
    return math.atan2(n2, n1)


def refractive_index(speed_in_medium: float) -> float:
    """Refractive index n = c / v from the speed of light in the medium."""
    return C / speed_in_medium


def numerical_aperture(n_core: float, n_clad: float) -> float:
    """Fibre numerical aperture NA = sqrt(n_core^2 - n_clad^2): the sine of the maximum
    half-angle of the cone of light the fibre will accept and guide by total internal
    reflection."""
    return math.sqrt(n_core * n_core - n_clad * n_clad)


def acceptance_angle(n_core: float, n_clad: float, n_outside: float = N_AIR) -> float:
    """Maximum half-angle (rad) of the acceptance cone: arcsin(NA / n_outside). Rays entering
    within this cone are trapped and guided down the fibre."""
    return math.asin(numerical_aperture(n_core, n_clad) / n_outside)

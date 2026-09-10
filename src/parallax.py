"""Parallax, proper motion, and space velocity: the geometry of stellar distance.

As Earth orbits the Sun, a nearby star appears to shift against the far background by
the parallax angle p. The distance follows from simple trigonometry, and the parsec is
defined so the arithmetic is trivial:

    d (parsecs) = 1 / p (arcseconds).

A star at 1 pc has a parallax of 1 arcsecond; the nearest star, Proxima, has p = 0.77"
(d = 1.30 pc). This is the first rung of the cosmic distance ladder, measured directly
by Hipparcos and Gaia for over a billion stars.

A star's velocity splits into two measurable pieces. The RADIAL velocity v_r (along the
line of sight) comes from the Doppler shift. The TANGENTIAL velocity v_t (across the
sky) is read from the proper motion mu -- the angular drift per year -- once the
distance is known:

    v_t = 4.74 mu d,     (v_t in km/s, mu in arcsec/yr, d in pc).

The 4.74 packs together the AU, the year and the arcsecond. The total SPACE VELOCITY is
then v = sqrt(v_r^2 + v_t^2). Barnard's Star, with the largest known proper motion
(10.4"/yr) at 1.83 pc, races across the sky at ~90 km/s tangential.

This module gives the distance from parallax, the tangential and space velocities from
proper motion and radial velocity, and the parallax expected at a distance, and
reproduces the 1 pc = 1" definition and Barnard's Star. SI-friendly astronomical units.
Pure stdlib; the astrometry companion to the distances and rotation-curve modules.
"""

from __future__ import annotations

import math

PC = 3.0856775814913673e16     # parsec in metres
AU = 1.495978707e11
YEAR = 3.15576e7
KM = 1e3

# v_t = k * mu * d with mu in arcsec/yr, d in pc, v_t in km/s
VT_CONST = 4.740470446         # = AU / (year) / (arcsec in rad) / 1000, km/s


def distance_pc(parallax_arcsec: float) -> float:
    """Distance in parsecs from parallax: d = 1 / p (p in arcseconds)."""
    return 1.0 / parallax_arcsec


def distance_ly(parallax_arcsec: float) -> float:
    """Distance in light-years (1 pc = 3.2616 ly)."""
    return distance_pc(parallax_arcsec) * 3.2615637


def parallax_at(distance_pc_val: float) -> float:
    """Parallax (arcseconds) a star at the given distance (pc) would show: p = 1/d."""
    return 1.0 / distance_pc_val


def tangential_velocity(proper_motion_asec_yr: float,
                        distance_pc_val: float) -> float:
    """Tangential (across-sky) velocity in km/s: v_t = 4.74 mu d, with mu in
    arcsec/yr and d in pc."""
    return VT_CONST * proper_motion_asec_yr * distance_pc_val


def space_velocity(radial_kms: float, proper_motion_asec_yr: float,
                   distance_pc_val: float) -> float:
    """Total space velocity in km/s: v = sqrt(v_r^2 + v_t^2)."""
    v_t = tangential_velocity(proper_motion_asec_yr, distance_pc_val)
    return math.hypot(radial_kms, v_t)


def proper_motion_from_vt(v_t_kms: float, distance_pc_val: float) -> float:
    """Invert to the proper motion (arcsec/yr) implied by a tangential velocity."""
    return v_t_kms / (VT_CONST * distance_pc_val)

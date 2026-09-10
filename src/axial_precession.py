"""Precession of the equinoxes: why the pole star changes over 26,000 years.

Earth is not a sphere but an oblate spheroid, bulging at the equator by about one
part in 300. Because the spin axis is tilted 23.4 degrees to the ecliptic, the Sun
and Moon pull unequally on the near and far sides of that equatorial bulge, exerting
a torque that tries to twist the equator into the orbital plane. A spinning body
responds to a torque perpendicular to its spin by precessing: the axis sweeps out a
cone instead of tipping over, exactly like a leaning gyroscope.

For a torque-driven top the axial precession rate is

    Omega_p = (3/2) (G M / r^3) (J2 / (1 + J2)) (I / (I C omega)) cos(eps)
            ~= (3/2) (n^2 / omega) (C - A)/C cos(eps),

where n^2 = G M / r^3 is the perturber's mean-motion (orbital angular frequency)
squared, omega is Earth's spin rate, eps the obliquity, and (C - A)/C the dynamical
oblateness (the "H" of ~0.00327). Summing the Sun and Moon contributions -- the Moon
dominates by roughly a factor of two because torque scales as M/r^3 -- gives a
luni-solar precession of about 50.3 arcseconds per year, a full 360-degree circuit
of the pole in about 25,800 years (the "Great Year").

This is why Polaris is only temporarily the North Star, why the tropical and sidereal
years differ by ~20 minutes, and why zodiac dates have slipped a whole sign since
antiquity. This module computes the single-body torque rate, the combined luni-solar
rate, and the precession period, and reproduces the ~25,800-year value. SI units;
angular rates convertible to arcsec/yr. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
M_SUN = 1.989e30
M_MOON = 7.342e22
AU = 1.495978707e11             # Earth-Sun distance (m)
R_MOON = 3.844e8               # Earth-Moon distance (m)

DAY = 86164.0905               # sidereal day (s): Earth's true rotation period
YEAR = 3.15576e7              # Julian year (s)
H_DYN = 0.0032737949          # dynamical ellipticity (C - A)/C of Earth
OBLIQUITY = math.radians(23.439291)

OMEGA_EARTH = 2.0 * math.pi / DAY   # spin angular rate (rad/s)

ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0


def torque_precession_rate(M: float, r: float, H: float = H_DYN,
                           eps: float = OBLIQUITY,
                           omega: float = OMEGA_EARTH) -> float:
    """Axial precession rate (rad/s) driven by one perturber of mass M at distance r:
    Omega_p = (3/2) (G M / r^3) (H / omega) cos(eps). Torque on the equatorial bulge
    scales as M / r^3, so the nearby Moon beats the far more massive Sun."""
    n2 = G * M / r ** 3
    return 1.5 * n2 * H / omega * math.cos(eps)


def lunisolar_rate(H: float = H_DYN, eps: float = OBLIQUITY,
                   omega: float = OMEGA_EARTH) -> float:
    """Combined Sun + Moon precession rate (rad/s)."""
    return (torque_precession_rate(M_SUN, AU, H, eps, omega)
            + torque_precession_rate(M_MOON, R_MOON, H, eps, omega))


def precession_period_years(rate_rad_s: float) -> float:
    """Time (years) for the axis to complete a 360-degree circuit at the given rate."""
    return (2.0 * math.pi / rate_rad_s) / YEAR


def rate_arcsec_per_year(rate_rad_s: float) -> float:
    """Convert an angular rate in rad/s to arcseconds per year."""
    return rate_rad_s * ARCSEC_PER_RAD * YEAR


def moon_to_sun_ratio() -> float:
    """Ratio of the Moon's to the Sun's torque contribution (~2.2): (M_moon/M_sun)
    (AU/r_moon)^3."""
    return (M_MOON / M_SUN) * (AU / R_MOON) ** 3

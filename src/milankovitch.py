"""Milankovitch cycles: how Earth's orbit paces the ice ages.

Earth's climate is nudged by slow changes in its orbit and spin, which redistribute
sunlight between seasons and latitudes even though the total annual sunshine barely moves.
Three cycles matter:

  ECCENTRICITY  e -- the orbit's ellipticity breathes between ~0.005 and ~0.06 over
                     ~100 kyr (and ~413 kyr), changing the Earth-Sun distance contrast.
  OBLIQUITY     eps -- the axial tilt wobbles between ~22.1 and ~24.5 degrees over ~41 kyr,
                     setting how strong the seasons are.
  PRECESSION    -- the tilt direction and the perihelion drift, so the season at which
                     Earth is closest to the Sun cycles over ~23/19 kyr; what counts for
                     ice sheets is the CLIMATIC PRECESSION e sin(omega).

Milankovitch's insight: Northern-Hemisphere high-latitude *summer* insolation governs
whether winter snow survives to build ice sheets. This module computes the daily insolation
at the top of the atmosphere from the standard astronomical formula

    H0 = arccos(-tan(phi) tan(delta))                 (sunrise hour angle)
    Q  = (S0/pi) (d0/d)^2 [H0 sin(phi) sin(delta) + cos(phi) cos(delta) sin(H0)]

with (d0/d)^2 = (1 + e cos(nu))^2 / (1 - e^2) the distance factor at true anomaly nu, and
the solar declination delta = arcsin(sin(eps) sin(lambda)) for solar longitude lambda. It
reproduces the June-solstice 65N peak of ~480-490 W/m^2 and the polar midnight-sun and
polar-night limits, and exposes the eccentricity / obliquity / climatic-precession knobs.
SI units, angles in radians unless _deg. Pure stdlib; the orbital-forcing companion to the
axial-precession and greenhouse notes.
"""

from __future__ import annotations

import math

S0 = 1361.0                      # solar constant (W/m^2)
OBLIQUITY_NOW = math.radians(23.44)


def declination(solar_longitude: float, obliquity: float = OBLIQUITY_NOW) -> float:
    """Solar declination delta = arcsin(sin(eps) sin(lambda)) (rad), where the solar
    longitude lambda is 0 at the March equinox and pi/2 at the June solstice."""
    return math.asin(math.sin(obliquity) * math.sin(solar_longitude))


def distance_factor(true_anomaly: float, e: float) -> float:
    """(d0/d)^2 = (1 + e cos nu)^2 / (1 - e^2): the ratio of the mean-distance solar flux
    to the flux at true anomaly nu, for an orbit of eccentricity e."""
    return (1.0 + e * math.cos(true_anomaly)) ** 2 / (1.0 - e * e)


def sunrise_hour_angle(latitude: float, decl: float) -> float:
    """Half-day-length hour angle H0 = arccos(-tan phi tan delta) (rad). Returns pi for the
    midnight-sun (sun never sets) and 0 for polar night (sun never rises)."""
    x = -math.tan(latitude) * math.tan(decl)
    if x <= -1.0:
        return math.pi          # polar day
    if x >= 1.0:
        return 0.0              # polar night
    return math.acos(x)


def daily_insolation(latitude: float, solar_longitude: float, e: float,
                     obliquity: float = OBLIQUITY_NOW, longitude_perihelion: float = 0.0,
                     s0: float = S0) -> float:
    """Mean daily top-of-atmosphere insolation (W/m^2) at latitude phi for solar longitude
    lambda, orbit (e, eps, omega=longitude of perihelion). The true anomaly is
    nu = lambda - omega."""
    decl = declination(solar_longitude, obliquity)
    nu = solar_longitude - longitude_perihelion
    dfac = distance_factor(nu, e)
    H0 = sunrise_hour_angle(latitude, decl)
    return (s0 / math.pi) * dfac * (
        H0 * math.sin(latitude) * math.sin(decl)
        + math.cos(latitude) * math.cos(decl) * math.sin(H0)
    )


def climatic_precession(e: float, longitude_perihelion: float) -> float:
    """The climatic precession parameter e sin(omega): the eccentricity-modulated precession
    that sets which season falls at perihelion. This, not raw precession, paces ice sheets."""
    return e * math.sin(longitude_perihelion)


def summer_solstice_insolation(latitude: float, e: float,
                               obliquity: float = OBLIQUITY_NOW,
                               longitude_perihelion: float = 0.0) -> float:
    """Northern-summer-solstice (lambda = pi/2) daily insolation (W/m^2) at latitude phi --
    the classic '65N June' Milankovitch target that governs ice-sheet growth."""
    return daily_insolation(latitude, math.pi / 2.0, e, obliquity, longitude_perihelion)

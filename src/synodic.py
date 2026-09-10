"""Synodic periods: how often planets line up.

A planet's SIDEREAL period is one orbit against the fixed stars. But what we actually
see from Earth -- an opposition, a conjunction, a launch window -- repeats on the
SYNODIC period, the time between successive identical Sun-Earth-planet alignments. Two
bodies with sidereal periods P1 and P2 catch up to the same relative geometry at the
beat frequency of their orbital rates:

    1 / S = | 1 / P1 - 1 / P2 |.

For Mars (P = 687 d) seen from Earth (365.25 d) this gives S ~ 780 days: Mars comes to
opposition, and the cheap launch window opens, only every ~26 months -- which is
exactly the cadence of Mars missions. For the Moon, the sidereal month (27.3 d, one
orbit against the stars) and the synodic month (29.5 d, new Moon to new Moon) differ
because the Earth-Moon system also orbits the Sun; the extra ~2.2 days is the synodic
correction.

Two objects with nearly equal periods (like co-orbital asteroids, or Earth and a
near-Earth object) have an enormous synodic period -- they drift past each other very
slowly -- while a fast inner planet laps a slow outer one often. This module gives the
synodic period from two sidereal periods, the relative (beat) angular rate, the number
of conjunctions per unit time, and reproduces the Mars 780-day window and the lunar
synodic month. Days as the natural unit. Pure stdlib; the alignment companion to the
Kepler and resonance modules.
"""

from __future__ import annotations

import math

DAY = 86400.0
YEAR_DAYS = 365.25

# sidereal orbital periods (days)
P_MERCURY = 87.969
P_VENUS = 224.701
P_EARTH = 365.256
P_MARS = 686.980
P_JUPITER = 4332.59
P_MOON_SIDEREAL = 27.3217


def synodic_period(P1: float, P2: float) -> float:
    """Synodic period S from two sidereal periods: 1/S = |1/P1 - 1/P2|. Same units in,
    same units out. Returns infinity for equal periods."""
    diff = abs(1.0 / P1 - 1.0 / P2)
    return float("inf") if diff == 0.0 else 1.0 / diff


def synodic_from_earth(P_planet: float, P_earth: float = P_EARTH) -> float:
    """Synodic period (days) of a planet as seen from Earth."""
    return synodic_period(P_planet, P_earth)


def relative_angular_rate(P1: float, P2: float) -> float:
    """Beat angular rate |n1 - n2| (radians per unit time) between two orbits."""
    return abs(2.0 * math.pi / P1 - 2.0 * math.pi / P2)


def conjunctions_per_year(P1: float, P2: float) -> float:
    """Number of conjunctions (identical alignments) per Earth year."""
    return YEAR_DAYS / synodic_period(P1, P2)


def synodic_month(P_moon_sid: float = P_MOON_SIDEREAL,
                  P_earth: float = P_EARTH) -> float:
    """Synodic (new-Moon-to-new-Moon) month from the sidereal month and Earth's orbit:
    1/S = 1/P_sid - 1/P_earth. ~29.5 days."""
    return synodic_period(P_moon_sid, P_earth)

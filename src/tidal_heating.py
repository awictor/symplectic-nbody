"""Tidal heating: why Io has volcanoes and Europa has an ocean.

A moon on an eccentric orbit is squeezed and stretched by its planet's tide by a
varying amount over each orbit. The moon is not perfectly elastic, so some of
that flexing energy is dissipated as HEAT. For a synchronously rotating moon the
orbit-averaged tidal heating rate is

    dE/dt = (21/2) (k2 / Q) (G M_p^2 R^5 n e^2) / a^6,

where M_p is the planet mass, R the moon's radius, a and e its orbital semi-major
axis and eccentricity, n = sqrt(G M_p / a^3) the mean motion, and k2/Q the tidal
Love-number-to-quality-factor ratio (the moon's response). The heating scales
steeply: as e^2, as R^5, and as a^{-15/2} (through n/a^6).

For Io this comes to ~1e14 W -- about 40 times Earth's total internal heat flux,
and the reason Io is the most volcanically active body in the Solar System. The
eccentricity that powers it is pumped by the Laplace 4:2:1 resonance with Europa
and Ganymede (see resonance.py), so tidal heating and orbital resonance are two
sides of the same story.

SI units. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
M_JUP = 1.898e27               # kg
# Io parameters
IO_R = 1.8216e6                # m
IO_A = 4.217e8                 # m (orbital semi-major axis)
IO_E = 0.0041                  # forced eccentricity (resonance-maintained)
IO_K2_OVER_Q = 0.015           # tidal response (k2 ~ 0.03, Q ~ 2 -> ~0.015)
IO_AREA = 4.0 * math.pi * IO_R ** 2


def mean_motion(M_planet: float, a: float) -> float:
    """Orbital mean motion n = sqrt(G M_planet / a^3)."""
    return math.sqrt(G * M_planet / a ** 3)


def tidal_heating_rate(M_planet: float, R: float, a: float, e: float,
                       k2_over_Q: float) -> float:
    """Orbit-averaged tidal heating power (watts):
    dE/dt = (21/2)(k2/Q) G M_p^2 R^5 n e^2 / a^6."""
    n = mean_motion(M_planet, a)
    return (21.0 / 2.0) * k2_over_Q * G * M_planet ** 2 * R ** 5 * n * e * e / a ** 6


def io_heating() -> float:
    """Tidal heating power of Io (watts)."""
    return tidal_heating_rate(M_JUP, IO_R, IO_A, IO_E, IO_K2_OVER_Q)


def surface_heat_flux(power: float, area: float) -> float:
    """Heat flux per unit area (W/m^2)."""
    return power / area

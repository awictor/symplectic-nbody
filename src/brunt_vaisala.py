"""The Brunt-Vaisala frequency: buoyancy, gravity waves, and convection.

Displace a fluid parcel upward in a stratified medium and one of two things happens.
If the parcel, expanding adiabatically as it rises into lower pressure, ends up denser
than its new surroundings, gravity pulls it back down -- and it overshoots, oscillating
about its equilibrium at the Brunt-Vaisala (buoyancy) frequency N. If instead the
risen parcel is lighter than its surroundings, buoyancy keeps pushing it up: the
stratification is unstable and convection sets in.

The buoyancy frequency is

    N^2 = g ( (1/gamma) dln p/dz - dln rho/dz )   (general),
        = (g/T)( dT/dz + g/c_p )                  (ideal gas, using potential temp),

where the second form compares the actual temperature gradient dT/dz to the adiabatic
lapse rate g/c_p. When N^2 > 0 the layer is stably stratified and supports internal
gravity waves of period 2 pi / N; when N^2 < 0 it is convectively unstable (the
Schwarzschild criterion). The sign of N^2 is exactly the buoyancy version of the
Schwarzschild convection criterion that decides how stars and atmospheres transport
heat.

For Earth's troposphere N ~ 0.01-0.02 s^-1 (buoyancy period ~5-10 min); the strongly
stratified stratosphere has larger N. In the Sun, g-modes live in the stably
stratified radiative core while the convective envelope has N^2 < 0.

This module gives N^2 and N for both an ideal gas (via the lapse rate) and a general
stratification, the buoyancy oscillation period, and the convective-stability verdict,
and reproduces the stable-troposphere / unstable-superadiabatic contrast. SI units.
Pure stdlib; the buoyancy companion to the atmosphere and opacity modules.
"""

from __future__ import annotations

import math

G_EARTH = 9.80665              # standard gravity (m/s^2)
CP_AIR = 1005.0                # specific heat of dry air at constant pressure (J/kg/K)


def adiabatic_lapse_rate(g: float = G_EARTH, c_p: float = CP_AIR) -> float:
    """Dry adiabatic lapse rate Gamma_d = g / c_p (K/m): how fast a rising parcel
    cools by adiabatic expansion (~9.8 K/km for Earth air)."""
    return g / c_p


def brunt_vaisala_squared_ideal(dT_dz: float, T: float, g: float = G_EARTH,
                                c_p: float = CP_AIR) -> float:
    """N^2 for an ideal gas (s^-2): N^2 = (g/T)(dT/dz + g/c_p). dT/dz is the actual
    (signed, usually negative) environmental temperature gradient. N^2>0 stable."""
    return (g / T) * (dT_dz + g / c_p)


def brunt_vaisala_squared_general(dlnp_dz: float, dlnrho_dz: float,
                                  g: float = G_EARTH,
                                  gamma: float = 1.4) -> float:
    """General N^2 (s^-2): N^2 = g( (1/gamma) dln p/dz - dln rho/dz ), from comparing
    the parcel's adiabatic density change to the environment's."""
    return g * (dlnp_dz / gamma - dlnrho_dz)


def brunt_vaisala_frequency(N2: float) -> float:
    """Buoyancy frequency N = sqrt(N^2) (rad/s) for a stable layer (N^2 > 0);
    returns 0 for unstable/neutral layers (N^2 <= 0)."""
    return math.sqrt(N2) if N2 > 0.0 else 0.0


def buoyancy_period(N2: float) -> float:
    """Oscillation period 2 pi / N (s) of a displaced parcel in a stable layer.
    Returns infinity for a non-oscillating (unstable/neutral) layer."""
    N = brunt_vaisala_frequency(N2)
    return 2.0 * math.pi / N if N > 0.0 else float("inf")


def is_convective(dT_dz: float, g: float = G_EARTH, c_p: float = CP_AIR) -> bool:
    """Schwarzschild criterion (ideal gas): convectively unstable when the actual
    temperature falls faster than the adiabatic lapse rate, i.e. N^2 < 0.
    dT/dz here is the signed environmental gradient (negative = cooling upward)."""
    return brunt_vaisala_squared_ideal(dT_dz, 300.0, g, c_p) < 0.0

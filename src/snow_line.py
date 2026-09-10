"""The snow line: why the inner solar system is rocky and the outer is icy.

A protoplanetary disk is heated by its star, so its temperature falls with distance.
For an optically-thin disk in radiative equilibrium a grain at distance r absorbs the
starlight it intercepts and re-radiates as a blackbody, giving

    T(r) = ( L_star / (16 pi sigma r^2) )^(1/4)  ~  T_star sqrt(R_star / (2 r)),

so T ~ r^(-1/2). The snow (frost) line is the distance where the disk cools past the
condensation temperature of water ice, ~150-170 K: inside it water is vapour and only
rock and metal condense; outside it ice condenses too, roughly tripling the density of
available solids.

That extra solid material is why the giant planets grew massive enough to seize gas
before the disk dissipated, and why the terrestrial planets inside ~2.7 AU stayed
small and dry. Different ices have different frost lines -- water near ~2.7 AU, CO2
and NH3 farther out, CO and N2 far beyond Neptune -- setting a whole sequence of
composition boundaries across the disk.

This module gives the equilibrium disk temperature, the snow-line distance for a
given condensation temperature, its scaling with stellar luminosity, and the solids-
density jump across the line, and reproduces the ~2.7 AU water snow line of the early
solar system. SI units. Pure stdlib; the disk-temperature companion to the blackbody
and habitable-zone modules.
"""

from __future__ import annotations

import math

SIGMA = 5.670374419e-8         # Stefan-Boltzmann constant (W m^-2 K^-4)
L_SUN = 3.828e26               # solar luminosity (W)
AU = 1.495978707e11

T_WATER_ICE = 160.0            # water-ice condensation temperature in a disk (K)
T_CO2_ICE = 70.0
T_CO_ICE = 20.0

# solids surface density roughly triples once water ice condenses
ICE_ENHANCEMENT = 3.0


def disk_temperature(r: float, L: float = L_SUN) -> float:
    """Radiative-equilibrium disk temperature (K) at distance r:
    T = (L / (16 pi sigma r^2))^(1/4), i.e. T ~ r^(-1/2)."""
    return (L / (16.0 * math.pi * SIGMA * r * r)) ** 0.25


def snow_line(T_cond: float = T_WATER_ICE, L: float = L_SUN) -> float:
    """Distance (m) at which the disk cools to the condensation temperature T_cond.
    Invert T(r): r = sqrt(L / (16 pi sigma T_cond^4))."""
    return math.sqrt(L / (16.0 * math.pi * SIGMA * T_cond ** 4))


def snow_line_au(T_cond: float = T_WATER_ICE, L: float = L_SUN) -> float:
    """Snow-line distance in astronomical units."""
    return snow_line(T_cond, L) / AU


def solids_density_jump(inside: float, factor: float = ICE_ENHANCEMENT) -> float:
    """Solids surface density just outside the water snow line: `factor` times the
    rock-only value `inside` (ice roughly triples the condensable material)."""
    return inside * factor


def condensation_temperature_at(r: float, L: float = L_SUN) -> float:
    """The ice whose frost line sits at distance r must condense at this temperature;
    just an alias for disk_temperature (a species condenses where T(r) = T_cond)."""
    return disk_temperature(r, L)

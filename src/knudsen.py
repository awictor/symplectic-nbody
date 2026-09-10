"""The Knudsen number: when a gas stops being a fluid.

Fluid dynamics treats a gas as a smooth continuum, but a gas is really a swarm of molecules
flying between collisions. That picture only holds when the mean free path lambda -- the
average distance a molecule travels before hitting another -- is tiny compared to the size L
of whatever you care about. Their ratio is the Knudsen number,

    Kn = lambda / L,        lambda = k_B T / (sqrt(2) pi d^2 P),

with d the molecular diameter and P the pressure. It sorts every gas flow into regimes:

    Kn < 0.01     continuum       -- ordinary Navier-Stokes, no-slip walls
    0.01 - 0.1    slip flow        -- Navier-Stokes but the gas slips at the wall
    0.1 - 10      transitional     -- neither picture alone works
    Kn > 10       free molecular   -- molecules fly wall-to-wall, collisions rare.

At sea level lambda ~ 68 nm, so Kn is minuscule for anything macroscopic and air is a
perfect fluid. But shrink L to a MEMS channel or a microchip's cooling pores, or raise the
altitude until the air thins (a satellite in the upper atmosphere sees free-molecular flow),
and Kn climbs until the continuum assumption fails -- gases slip at walls, thermal creep sets
in, and drag must be computed molecule by molecule.

This module gives the mean free path, the Knudsen number, the flow-regime classification,
the pressure or size at which a target Kn is reached, and the mean molecular speed, and
reproduces air's ~68 nm sea-level mean free path and the continuum-to-molecular transition.
SI units. Pure stdlib; the rarefied-gas companion to the kinetic-theory and Reynolds notes.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23             # Boltzmann constant (J/K)
D_AIR = 3.7e-10               # effective air-molecule diameter (m)
M_AIR = 4.81e-26             # mean mass of an air molecule (kg)


def mean_free_path(temperature: float, pressure: float, diameter: float = D_AIR) -> float:
    """Mean free path lambda = k_B T / (sqrt(2) pi d^2 P) (m): the average distance a
    molecule travels between collisions. ~68 nm for air at sea level."""
    return K_B * temperature / (math.sqrt(2.0) * math.pi * diameter * diameter * pressure)


def knudsen_number(length: float, temperature: float, pressure: float,
                   diameter: float = D_AIR) -> float:
    """Knudsen number Kn = lambda / L: the mean free path over the characteristic size L.
    Small = continuum (fluid), large = free molecular (ballistic molecules)."""
    return mean_free_path(temperature, pressure, diameter) / length


def flow_regime(kn: float) -> str:
    """Classify a gas flow by Knudsen number: 'continuum' (<0.01), 'slip' (0.01-0.1),
    'transitional' (0.1-10), or 'free molecular' (>10)."""
    if kn < 0.01:
        return "continuum"
    if kn < 0.1:
        return "slip"
    if kn < 10.0:
        return "transitional"
    return "free molecular"


def is_continuum(kn: float) -> bool:
    """True if the continuum (Navier-Stokes, no-slip) description holds, Kn < 0.01."""
    return kn < 0.01


def pressure_for_knudsen(kn_target: float, length: float, temperature: float,
                         diameter: float = D_AIR) -> float:
    """Pressure (Pa) at which a gas in a system of size L reaches a target Knudsen number:
    P = k_B T / (sqrt(2) pi d^2 Kn L). Below it the gas is more rarefied."""
    return K_B * temperature / (math.sqrt(2.0) * math.pi * diameter * diameter * kn_target * length)


def length_for_knudsen(kn_target: float, temperature: float, pressure: float,
                       diameter: float = D_AIR) -> float:
    """Characteristic size (m) at which a gas reaches a target Knudsen number:
    L = lambda / Kn. Below this size the continuum picture breaks down."""
    return mean_free_path(temperature, pressure, diameter) / kn_target


def mean_speed(temperature: float, mass: float = M_AIR) -> float:
    """Mean molecular speed sqrt(8 k_B T / (pi m)) (m/s): ~468 m/s for air at 300 K."""
    return math.sqrt(8.0 * K_B * temperature / (math.pi * mass))

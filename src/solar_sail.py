"""Solar sails and radiation pressure: pushing spacecraft with sunlight.

Light carries momentum. A surface absorbing a flux F feels a pressure F/c; a perfectly
reflecting one feels twice that, 2F/c, because the photons reverse. At Earth's distance
the solar flux is ~1361 W/m^2, so a mirror sail feels only ~9 micropascals -- a feather
touch, but one that never runs out of propellant and, integrated over months, builds
enormous velocity.

Because both sunlight and gravity fall as 1/r^2, the ratio of radiation force to solar
gravity is a distance-independent number, the lightness number

    beta = F_rad / F_grav = (2 L Q A) / (4 pi c G M_sun m),

for a sail of area A and mass m (Q the reflectivity efficiency). beta = 1 means the sail
feels an outward push exactly cancelling the Sun's pull -- it coasts in a straight line;
beta > 1 lets it spiral outward or even leave the Solar System. The critical
area-to-mass ratio for beta = 1 is ~1.5 g/m^2, which is why real sails (IKAROS,
LightSail 2) must be gossamer-thin.

This module gives the radiation pressure, the force and acceleration on a sail, the
lightness number beta, and the area-to-mass ratio for a target beta, and reproduces the
9-micropascal pressure at 1 AU and the ~1.5 g/m^2 critical loading. SI units. Pure
stdlib; the photon-momentum companion to the Poynting-Robertson and Eddington modules.
"""

from __future__ import annotations

import math

C = 2.99792458e8
G = 6.67430e-11
L_SUN = 3.828e26
M_SUN = 1.989e30
AU = 1.495978707e11

# solar irradiance at 1 AU (W/m^2)
SOLAR_CONSTANT = 1361.0


def solar_flux(r: float) -> float:
    """Solar flux (W/m^2) at distance r: L_sun / (4 pi r^2)."""
    return L_SUN / (4.0 * math.pi * r * r)


def radiation_pressure(r: float, reflectivity: float = 1.0) -> float:
    """Radiation pressure (Pa) on a surface at distance r. reflectivity=0 (black)
    gives F/c; =1 (perfect mirror) gives 2F/c."""
    return (1.0 + reflectivity) * solar_flux(r) / C


def sail_force(area: float, r: float, reflectivity: float = 1.0) -> float:
    """Force (N) on a sail of area A facing the Sun at distance r."""
    return radiation_pressure(r, reflectivity) * area


def sail_acceleration(area: float, mass: float, r: float,
                      reflectivity: float = 1.0) -> float:
    """Acceleration (m/s^2) of a sail of area A and total mass m at distance r."""
    return sail_force(area, r, reflectivity) / mass


def lightness_number(area: float, mass: float, reflectivity: float = 1.0) -> float:
    """Lightness number beta = radiation force / solar gravity (distance-independent):
    beta = (1+reflectivity) L_sun A / (4 pi c G M_sun m). beta=1 cancels gravity."""
    sigma = area / mass
    return (1.0 + reflectivity) * L_SUN * sigma / (4.0 * math.pi * C * G * M_SUN)


def critical_area_to_mass(reflectivity: float = 1.0) -> float:
    """Area-to-mass ratio (m^2/kg) giving beta = 1 (radiation exactly cancels the
    Sun's gravity). ~0.77 m^2/kg for a black sail, ~1.5 g/m^2 loading for a mirror."""
    return 4.0 * math.pi * C * G * M_SUN / ((1.0 + reflectivity) * L_SUN)

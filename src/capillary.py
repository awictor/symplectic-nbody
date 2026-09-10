"""The capillary length: where surface tension loses to gravity.

Why is a dewdrop a near-perfect sphere while a puddle is a flat sheet? Both are the same
liquid; what differs is size. Surface tension gamma pulls a blob toward the minimum-area
sphere, while gravity flattens anything tall enough that its own weight matters. The two
balance at a single length scale, the capillary length:

    l_c = sqrt(gamma / (rho g)),

about 2.7 mm for water. Below l_c surface tension wins and drops are round; above it gravity
wins and liquid puddles out to a film whose thickness is itself ~2 l_c. The dimensionless
ratio of the two effects over a body of size L is the Bond number (a.k.a. Eotvos number):

    Bo = rho g L^2 / gamma = (L / l_c)^2,

so Bo < 1 is the surface-tension regime and Bo > 1 the gravity regime. When a drop is
*moving* through or against another fluid, inertia enters too, measured by the Weber number

    We = rho v^2 L / gamma,

the ratio of disrupting inertial pressure to the restoring surface tension; a droplet breaks
up once We exceeds a critical ~12. The same gamma sets the maximum height a static drop can
pile to (~2 l_c) and the fastest capillary (Rayleigh-Plateau) breakup of a thin jet into
drops spaced ~9 radii apart.

This module gives the capillary length, the Bond and Weber numbers and the size or velocity
at which each equals one, the maximum puddle depth, and the Rayleigh-Plateau drop spacing,
and reproduces water's 2.7 mm capillary length and the We ~ 12 breakup threshold. SI units.
Pure stdlib; the interface-scaling companion to the surface-tension and Bernoulli notes.
"""

from __future__ import annotations

import math

G_EARTH = 9.80665
RHO_WATER = 998.0
GAMMA_WATER = 0.0728          # N/m at 20 C
WE_CRIT = 12.0                # critical Weber number for droplet breakup


def capillary_length(gamma: float, rho: float = RHO_WATER, g: float = G_EARTH) -> float:
    """Capillary length l_c = sqrt(gamma / (rho g)) (m): the crossover size between
    surface-tension-dominated (below) and gravity-dominated (above). ~2.7 mm for water."""
    return math.sqrt(gamma / (rho * g))


def bond_number(length: float, gamma: float, rho: float = RHO_WATER,
                g: float = G_EARTH) -> float:
    """Bond (Eotvos) number Bo = rho g L^2 / gamma = (L/l_c)^2. Bo < 1: surface tension
    dominates (round drops); Bo > 1: gravity dominates (flat puddles)."""
    return rho * g * length * length / gamma


def weber_number(velocity: float, length: float, gamma: float,
                 rho: float = RHO_WATER) -> float:
    """Weber number We = rho v^2 L / gamma: disrupting inertial pressure over restoring
    surface tension. A moving drop breaks up once We exceeds ~12."""
    return rho * velocity * velocity * length / gamma


def droplet_breaks_up(velocity: float, length: float, gamma: float,
                      rho: float = RHO_WATER, we_crit: float = WE_CRIT) -> bool:
    """True if a drop of size L moving at velocity v through a fluid of density rho exceeds
    the critical Weber number and fragments."""
    return weber_number(velocity, length, gamma, rho) > we_crit


def breakup_velocity(length: float, gamma: float, rho: float = RHO_WATER,
                     we_crit: float = WE_CRIT) -> float:
    """Relative velocity at which a drop of size L reaches the critical Weber number:
    v = sqrt(We_crit gamma / (rho L)) (m/s)."""
    return math.sqrt(we_crit * gamma / (rho * length))


def max_puddle_depth(gamma: float, rho: float = RHO_WATER, g: float = G_EARTH,
                     contact_angle_rad: float = math.pi) -> float:
    """Maximum depth (m) of a static puddle of a non-wetting liquid on a flat surface:
    h = 2 l_c sin(theta/2). For a fully non-wetting drop (theta = 180 deg) this is 2 l_c
    (~5.4 mm for water); it shrinks toward zero as the liquid wets the surface."""
    return 2.0 * capillary_length(gamma, rho, g) * math.sin(contact_angle_rad / 2.0)


def rayleigh_plateau_spacing(radius: float) -> float:
    """Wavelength of the fastest-growing Rayleigh-Plateau instability on a liquid cylinder
    of given radius: lambda = 2 pi radius / 0.697 ~ 9.01 radius. A thin jet pinches into
    drops at roughly this spacing."""
    return 2.0 * math.pi * radius / 0.697

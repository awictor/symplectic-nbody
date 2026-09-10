"""Free-fall and dynamical timescales: the clock every self-gravitating system runs on.

Take away a cloud's pressure support and it collapses under its own gravity. The
time for a uniform sphere of mean density rho to collapse to a point -- the free-fall
time -- depends only on that density,

    t_ff = sqrt( 3 pi / (32 G rho) ),

with no reference to the cloud's size or mass. The closely related dynamical
(gravitational) time and the orbital period at the edge share the same 1/sqrt(G rho)
scaling, so a single number sets how fast anything gravitational happens:

    t_dyn ~ 1 / sqrt(G rho),   P_orbit = 2 pi / sqrt(G rho_mean) (for a test particle
    skimming a body of mean density rho).

This is why every gravitational system, from a collapsing molecular cloud to a galaxy,
"ticks" on its free-fall time: denser things evolve faster. A molecular-cloud core
(n ~ 10^4 cm^-3) collapses in a few hundred thousand years; the Sun (mean density
~1400 kg/m^3) would free-fall in under an hour if fusion switched off; the Earth in a
few minutes. The free-fall time also caps how fast a star can shine on gravity alone
(the Kelvin-Helmholtz time is longer only because pressure resists).

This module gives the free-fall time, the dynamical time, and the mean-density orbital
period, their universal 1/sqrt(rho) scaling, and the mean density of a body, and
reproduces the Sun's ~30-minute free-fall time. SI units. Pure stdlib; the timescale
that underlies the Jeans-collapse and star-formation modules.
"""

from __future__ import annotations

import math

G = 6.67430e-11
M_SUN = 1.989e30
R_SUN = 6.957e8
MYR = 3.15576e13
MINUTE = 60.0


def free_fall_time(rho: float) -> float:
    """Free-fall collapse time (s) of a uniform sphere of mean density rho:
    t_ff = sqrt(3 pi / (32 G rho)). Independent of size and mass."""
    return math.sqrt(3.0 * math.pi / (32.0 * G * rho))


def dynamical_time(rho: float) -> float:
    """Dynamical / gravitational time t_dyn = 1 / sqrt(G rho) (s): the same
    1/sqrt(G rho) clock, without the geometric 3 pi/32 factor."""
    return 1.0 / math.sqrt(G * rho)


def mean_density(M: float, R: float) -> float:
    """Mean density (kg/m^3) of a body of mass M and radius R: M / (4/3 pi R^3)."""
    return M / (4.0 / 3.0 * math.pi * R ** 3)


def orbital_period_mean_density(rho: float) -> float:
    """Orbital period (s) of a test particle skimming the surface of a body of mean
    density rho: P = sqrt(3 pi / (G rho)) = 2 pi / sqrt(G rho_mean * 4/3 pi ... );
    P = sqrt(3 pi / (G rho)). Depends only on mean density -- Kepler's third law in
    disguise, and why all low orbits have ~90-minute periods around Earth."""
    return math.sqrt(3.0 * math.pi / (G * rho))


def free_fall_time_from_body(M: float, R: float) -> float:
    """Free-fall time (s) for a body specified by its mass and radius."""
    return free_fall_time(mean_density(M, R))

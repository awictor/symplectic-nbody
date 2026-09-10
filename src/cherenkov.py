"""Cherenkov radiation: the blue glow of going faster than light.

Nothing beats light in vacuum, but light slows to c/n in a medium of refractive index n, and
a charged particle can outrun *that*. When it does, it drags a shock front of electromagnetic
waves behind it -- the optical analogue of a sonic boom -- radiating a faint blue cone. This
is Cherenkov radiation, the eerie glow of a reactor core underwater and the signal that
detectors like IceCube and Super-Kamiokande use to catch neutrinos.

Emission happens only above a threshold speed, when the particle is superluminal in the
medium:

    beta > 1/n        (beta = v/c),

and the light comes out on a cone whose half-angle is set, exactly as for a Mach cone, by the
ratio of the wave speed to the particle speed:

    cos(theta_c) = 1 / (n beta).

So the cone opens up as the particle speeds toward beta = 1, saturating at a maximum angle
arccos(1/n). Measuring theta_c gives the particle's velocity, and combined with its momentum,
its mass -- which is how ring-imaging Cherenkov (RICH) detectors identify particles. The
threshold in terms of Lorentz factor is gamma_thr = 1/sqrt(1 - 1/n^2), and the number of
photons radiated per unit path (the Frank-Tamm result) grows with sin^2(theta_c).

This module gives the Cherenkov threshold speed and Lorentz factor, the cone half-angle, the
maximum angle, the velocity inferred from a measured cone, and a threshold test, and
reproduces water's ~0.75 beta threshold and the ~41-degree maximum cone. SI-free (beta and n
are dimensionless; angles in radians unless _deg). Pure stdlib; the relativistic-radiation
companion to the Mach-cone and synchrotron notes.
"""

from __future__ import annotations

import math


def threshold_beta(n: float) -> float:
    """Minimum speed beta = v/c for Cherenkov emission in a medium of index n: beta > 1/n.
    Water (n=1.33) -> 0.752; a particle must exceed this to glow."""
    return 1.0 / n


def threshold_gamma(n: float) -> float:
    """Threshold Lorentz factor gamma = 1/sqrt(1 - 1/n^2) for Cherenkov emission -- the
    minimum energy (in units of rest mass) a particle needs to radiate in the medium."""
    b = threshold_beta(n)
    return 1.0 / math.sqrt(1.0 - b * b)


def emits(beta: float, n: float) -> bool:
    """True if a particle at speed beta radiates Cherenkov light in a medium of index n
    (beta > 1/n, i.e. faster than light in the medium)."""
    return beta > threshold_beta(n)


def cone_angle(beta: float, n: float) -> float:
    """Cherenkov cone half-angle theta_c = arccos(1/(n beta)) (rad). Raises below threshold
    (no cone). Opens toward the maximum as beta -> 1."""
    x = 1.0 / (n * beta)
    if x > 1.0:
        raise ValueError("below Cherenkov threshold: no emission")
    return math.acos(x)


def max_cone_angle(n: float) -> float:
    """Maximum cone half-angle arccos(1/n) (rad), reached as beta -> 1. ~41 deg for water."""
    return math.acos(1.0 / n)


def velocity_from_cone(theta_c: float, n: float) -> float:
    """Particle speed beta = 1/(n cos(theta_c)) inferred from a measured Cherenkov cone
    angle -- how RICH detectors read off velocity. Inverts cone_angle."""
    return 1.0 / (n * math.cos(theta_c))


def photon_yield_factor(beta: float, n: float) -> float:
    """Relative Frank-Tamm photon yield per unit path, proportional to sin^2(theta_c) =
    1 - 1/(n beta)^2. Zero at threshold, rising toward 1 - 1/n^2 as beta -> 1."""
    if not emits(beta, n):
        return 0.0
    return 1.0 - 1.0 / (n * beta) ** 2

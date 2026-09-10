"""Fermi acceleration: how shocks manufacture cosmic rays with a universal spectrum.

Enrico Fermi asked how cosmic rays reach such enormous energies and found the answer
in repeated small kicks. A charged particle bouncing between converging magnetic
mirrors -- or across a shock front -- gains a little energy each round trip and has a
fixed chance of escaping. That combination of "multiply the energy by a constant
factor per cycle, lose a constant fraction of particles per cycle" produces a
power-law energy spectrum, N(E) dE ~ E^(-p) dE, with no preferred scale.

There are two versions. In *second-order* (original) Fermi acceleration a particle
scatters off randomly moving magnetic clouds; head-on collisions (energy gain) are
slightly more frequent than overtaking ones (energy loss), so the mean gain per
cycle is second order in the cloud speed, ~(V/c)^2 -- slow. In *first-order*
(diffusive shock) acceleration the particle repeatedly crosses a shock, always seeing
converging flow from both sides, so every crossing gains energy at first order in the
shock speed, ~(V/c) -- much faster, and it is the workhorse for supernova-remnant
cosmic rays.

The beautiful result of diffusive shock acceleration is that the spectral index
depends only on the shock compression ratio r = rho2/rho1:

    p = (r + 2) / (r - 1).

For a strong shock r -> 4 (the strong-shock limit of the Rankine-Hugoniot jump for a
monatomic gas), giving p = 2 -- the near-universal E^(-2) spectrum seen in
supernova remnants and inferred for the sources of Galactic cosmic rays.

This module gives the compression ratio, the spectral index, the per-cycle energy
gain and escape probability, and the resulting power law, and reproduces the p = 2
strong-shock result. SI units where relevant; spectra are dimensionless indices.
Pure stdlib; the particle-acceleration companion to the Sedov blast-wave module.
"""

from __future__ import annotations

import math


def compression_ratio(mach: float, gamma: float = 5.0 / 3.0) -> float:
    """Rankine-Hugoniot density compression across a shock of sonic Mach number M:
    r = (gamma+1) M^2 / ((gamma-1) M^2 + 2). r -> (gamma+1)/(gamma-1) = 4 as M -> inf."""
    m2 = mach * mach
    return (gamma + 1.0) * m2 / ((gamma - 1.0) * m2 + 2.0)


def spectral_index(r: float) -> float:
    """Differential spectral index p in N(E) ~ E^(-p) from diffusive shock
    acceleration: p = (r + 2) / (r - 1). Strong shock r=4 -> p=2."""
    return (r + 2.0) / (r - 1.0)


def spectral_index_from_mach(mach: float, gamma: float = 5.0 / 3.0) -> float:
    """Spectral index directly from the shock Mach number."""
    return spectral_index(compression_ratio(mach, gamma))


def energy_gain_per_cycle(beta: float, order: int = 1) -> float:
    """Fractional energy gain per acceleration cycle. order=1 (diffusive shock):
    <dE/E> ~ (4/3) beta, first order in beta = V/c. order=2 (cloud scattering):
    <dE/E> ~ (4/3) beta^2, second order (much smaller)."""
    if order == 1:
        return 4.0 / 3.0 * beta
    return 4.0 / 3.0 * beta * beta


def escape_probability(u2: float, c: float = 2.99792458e8) -> float:
    """Probability a relativistic particle is swept downstream and escapes per cycle:
    P_esc = 4 u2 / c, with u2 the downstream flow speed in the shock frame."""
    return 4.0 * u2 / c


def power_law(E: float, E0: float, p: float) -> float:
    """Differential number spectrum N(E) = (E/E0)^(-p) (normalized to 1 at E0)."""
    return (E / E0) ** (-p)


def strong_shock_index(gamma: float = 5.0 / 3.0) -> float:
    """Spectral index in the strong-shock limit (M -> infinity): r=(gamma+1)/(gamma-1),
    p=(r+2)/(r-1). For gamma=5/3, r=4 and p=2."""
    r = (gamma + 1.0) / (gamma - 1.0)
    return spectral_index(r)

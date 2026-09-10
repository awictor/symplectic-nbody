"""The Penrose process: extracting energy from a spinning black hole.

Inside a Kerr black hole's ergosphere, an object can have NEGATIVE energy as seen
from infinity. If a body splits there and the negative-energy fragment falls
through the horizon, the escaping fragment carries out MORE energy than went in --
mined from the hole's rotation. The hole spins down, and this continues until it
is a non-rotating Schwarzschild hole.

The bookkeeping is the irreducible mass -- the mass of the Schwarzschild hole you
are left with, which can never decrease (it is c^2/(16 pi G) times the horizon
area, Hawking's area theorem):

    M_irr = sqrt( ( M + sqrt(M^2 - a^2) ) / 2 )     (units G = c = 1, a = J/M).

The ROTATIONAL energy available to extract is

    E_rot = (M - M_irr) c^2,

up to 29% of the total mass-energy for an extremal (a = M) hole. The maximum
efficiency of a single idealized Penrose scattering is

    eta_max = 1 - 1/sqrt(2) ~ 20.7%   (extremal hole),

the famous Wald limit. This module computes the irreducible mass, the extractable
rotational energy, and the per-event efficiency. Units of M unless noted. Pure
stdlib; reuses the Kerr horizon.
"""

from __future__ import annotations

import math

from kerr import horizons


def irreducible_mass(M: float, a: float) -> float:
    """M_irr = sqrt((M + sqrt(M^2 - a^2))/2). The Schwarzschild-equivalent mass
    left after all rotational energy is extracted; never decreases."""
    return math.sqrt((M + math.sqrt(M * M - a * a)) / 2.0)


def rotational_energy(M: float, a: float) -> float:
    """Extractable rotational energy (M - M_irr), in units of M c^2 when M=1."""
    return M - irreducible_mass(M, a)


def rotational_energy_fraction(M: float, a: float) -> float:
    """Fraction of the total mass-energy that is rotational (extractable)."""
    return rotational_energy(M, a) / M


def max_efficiency_extremal() -> float:
    """Maximum single-scatter Penrose efficiency for an extremal hole,
    eta = 1 - 1/sqrt(2) ~ 0.2929 (mass-energy fraction extractable), and the
    per-event Wald bound 1 - 1/sqrt(2) ~ 20.7% is the classic figure. We return
    the full extractable fraction for a=M (0.2929)."""
    return rotational_energy_fraction(1.0, 1.0)


def area_irreducible(M: float, a: float) -> float:
    """Horizon area in units where it equals 16 pi M_irr^2 (so it tracks M_irr^2
    and can only grow -- the area theorem)."""
    return 16.0 * math.pi * irreducible_mass(M, a) ** 2

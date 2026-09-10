"""Fermi degeneracy pressure: the quantum pressure that holds up dead stars.

The Pauli exclusion principle forbids two electrons in the same state, so a cold,
dense electron gas resists compression even at zero temperature -- degeneracy
pressure. It is what supports white dwarfs against gravity (and, for neutrons,
neutron stars). The physics under the Chandrasekhar mass and the TOV equation.

Filling momentum states up to the Fermi momentum p_F sets everything:

    p_F = h / 2 * (3 n / pi)^{1/3},         (n = number density)
    E_F = p_F^2 / (2 m)                     (non-relativistic Fermi energy).

The pressure has two regimes:

    non-relativistic (p_F << m c):  P = (h^2 / 20 m) (3/pi)^{2/3} n^{5/3},
    ultra-relativistic (p_F >> m c): P = (hc / 8) (3/pi)^{1/3} n^{4/3}.

The softening of the exponent from 5/3 to 4/3 as electrons turn relativistic is
exactly why a maximum white-dwarf mass exists. This module gives the Fermi
momentum/energy and both pressure laws, and locates the relativistic transition.
SI units. Pure stdlib.
"""

from __future__ import annotations

import math

H = 6.62607015e-34
HBAR = 1.054571817e-34
C = 2.99792458e8
M_E = 9.1093837e-31
M_P = 1.6726219e-27


def fermi_momentum(n: float) -> float:
    """Fermi momentum p_F = (h/2)(3 n / pi)^{1/3} for number density n."""
    return 0.5 * H * (3.0 * n / math.pi) ** (1.0 / 3.0)


def fermi_energy(n: float, m: float = M_E) -> float:
    """Non-relativistic Fermi energy E_F = p_F^2 / (2 m)."""
    pF = fermi_momentum(n)
    return pF * pF / (2.0 * m)


def pressure_nonrel(n: float, m: float = M_E) -> float:
    """Non-relativistic degeneracy pressure P = (h^2/20 m)(3/pi)^{2/3} n^{5/3}."""
    return (H * H / (20.0 * m)) * (3.0 / math.pi) ** (2.0 / 3.0) * n ** (5.0 / 3.0)


def pressure_relativistic(n: float) -> float:
    """Ultra-relativistic degeneracy pressure P = (h c / 8)(3/pi)^{1/3} n^{4/3}."""
    return (H * C / 8.0) * (3.0 / math.pi) ** (1.0 / 3.0) * n ** (4.0 / 3.0)


def is_relativistic(n: float, m: float = M_E) -> bool:
    """True if the Fermi momentum exceeds m c (electrons are relativistic)."""
    return fermi_momentum(n) > m * C


def transition_density(m: float = M_E) -> float:
    """Number density where p_F = m c (the non-rel -> rel transition):
    n = (pi / 3)(2 m c / h)^3."""
    return (math.pi / 3.0) * (2.0 * m * C / H) ** 3

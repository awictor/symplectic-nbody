"""Quantum tunneling: passing through a barrier you cannot climb.

Classically a particle with energy E < V cannot cross a potential barrier of height V.
Quantum-mechanically its wavefunction leaks in as a decaying exponential, so there is a
finite probability of appearing on the far side -- tunneling.

For a rectangular barrier of height V and width L, in the thick-barrier limit the
transmission probability is

    T ~ exp(-2 kappa L),      kappa = sqrt(2 m (V - E)) / hbar,

falling exponentially with the width and with the square root of the energy deficit.
The exact expression adds a prefactor, but the exponential dominates: a barrier a few
decay-lengths thick is essentially opaque, yet thinning it or lowering it a little sends
the transmission soaring. That razor sensitivity is exactly what the scanning tunneling
microscope exploits -- the tunneling current between tip and surface changes by an order
of magnitude for every ~0.1 nm of gap, so it maps atoms.

The same physics runs alpha decay (an alpha particle tunneling out of the nuclear
Coulomb barrier, the Gamow theory) and fusion (nuclei tunneling in). This module gives
the decay constant kappa, the rectangular-barrier transmission (exact and thick-limit),
and the WKB transmission through a general barrier, and reproduces the exponential
STM-gap sensitivity. SI units, energies via an eV helper. Pure stdlib; the barrier-
penetration companion to the Gamow and uncertainty modules.
"""

from __future__ import annotations

import math

HBAR = 1.054571817e-34
M_E = 9.1093837015e-31
EV = 1.602176634e-19


def decay_constant(V_minus_E: float, m: float = M_E) -> float:
    """Wavefunction decay constant inside the barrier kappa = sqrt(2 m (V-E)) / hbar
    (1/m). V_minus_E is the energy deficit V - E in joules."""
    return math.sqrt(2.0 * m * V_minus_E) / HBAR


def transmission_thick(V_minus_E: float, L: float, m: float = M_E) -> float:
    """Thick-barrier transmission probability T ~ exp(-2 kappa L)."""
    kappa = decay_constant(V_minus_E, m)
    return math.exp(-2.0 * kappa * L)


def transmission_exact(E: float, V: float, L: float, m: float = M_E) -> float:
    """Exact rectangular-barrier transmission for E < V:
    T = 1 / (1 + V^2 sinh^2(kappa L) / (4 E (V - E)))."""
    if E >= V:
        return 1.0
    kappa = decay_constant(V - E, m)
    s = math.sinh(kappa * L)
    return 1.0 / (1.0 + V * V * s * s / (4.0 * E * (V - E)))


def wkb_transmission(barrier_func, x0: float, x1: float, E: float,
                     m: float = M_E, steps: int = 2000) -> float:
    """WKB transmission through a general barrier V(x) between the classical turning
    points x0, x1: T ~ exp(-2 integral sqrt(2 m (V(x) - E))/hbar dx), by the trapezoid
    rule over the classically forbidden region where V(x) > E."""
    dx = (x1 - x0) / steps
    integral = 0.0
    prev = None
    for i in range(steps + 1):
        x = x0 + i * dx
        arg = 2.0 * m * (barrier_func(x) - E)
        cur = math.sqrt(arg) / HBAR if arg > 0.0 else 0.0
        if prev is not None:
            integral += 0.5 * (prev + cur) * dx
        prev = cur
    return math.exp(-2.0 * integral)


def stm_current_ratio(gap1: float, gap2: float, work_function_ev: float = 4.0) -> float:
    """Ratio of STM tunneling currents at two tip-surface gaps (I ~ T ~ exp(-2 kappa d)),
    with the metal work function as the effective barrier height. Shows the ~order-of-
    magnitude-per-Angstrom sensitivity."""
    kappa = decay_constant(work_function_ev * EV)
    return math.exp(-2.0 * kappa * (gap2 - gap1))

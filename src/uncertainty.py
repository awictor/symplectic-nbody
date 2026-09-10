"""The Heisenberg uncertainty principle: why quantum things cannot sit still.

Position and momentum cannot both be sharply defined. Their spreads obey

    dx dp >= hbar / 2,

and the same relation holds for energy and time, dE dt >= hbar/2. This is not a
measurement clumsiness but a fundamental property of wave-like matter: confining a
particle to a small region dx forces a large momentum spread dp >= hbar/(2 dx), and
hence an irreducible kinetic energy.

That "zero-point" energy has real consequences. Confining an electron to an atom-sized
box (~0.1 nm) forces a momentum spread that gives a kinetic energy of a few eV -- the
scale of atomic binding, and the reason electrons do not spiral into the nucleus.
Confining a nucleon to a nucleus (~few fm) demands MeV-scale momenta, the scale of
nuclear energies. Estimating the ground-state energy by minimizing the confinement
kinetic energy plus the potential reproduces the hydrogen binding energy and the Bohr
radius from the uncertainty principle alone.

This module gives the minimum momentum/position/energy spreads, the confinement (zero-
point) kinetic energy of a particle in a box, and the energy-time bound (a line's
natural width from its lifetime), and reproduces the eV atomic and MeV nuclear scales.
SI units, energies via an eV/MeV helper. Pure stdlib; the quantum-limit companion to
the de-Broglie and Bohr modules.
"""

from __future__ import annotations

import math

HBAR = 1.054571817e-34
H = 6.62607015e-34
M_E = 9.1093837015e-31
M_P = 1.67262192e-27
EV = 1.602176634e-19
MEV = 1e6 * EV
FM = 1e-15


def min_momentum_spread(dx: float) -> float:
    """Minimum momentum spread dp = hbar / (2 dx) (kg m/s) for a position spread dx."""
    return HBAR / (2.0 * dx)


def min_position_spread(dp: float) -> float:
    """Minimum position spread dx = hbar / (2 dp) (m) for a momentum spread dp."""
    return HBAR / (2.0 * dp)


def confinement_energy(dx: float, m: float = M_E) -> float:
    """Zero-point (confinement) kinetic energy of a particle localized to size dx:
    E ~ dp^2 / (2m) = hbar^2 / (8 m dx^2). The irreducible energy of confinement."""
    dp = min_momentum_spread(dx)
    return dp * dp / (2.0 * m)


def energy_time_bound(dt: float) -> float:
    """Minimum energy spread dE = hbar / (2 dt) (J) for a state living a time dt.
    A short-lived state has a broad (Lorentzian) energy/line width."""
    return HBAR / (2.0 * dt)


def natural_linewidth_hz(lifetime: float) -> float:
    """Natural line width (Hz) of a transition with excited-state lifetime `lifetime`:
    dnu = 1 / (2 pi lifetime), the energy-time bound expressed as a frequency."""
    return 1.0 / (2.0 * math.pi * lifetime)


def hydrogen_ground_state_estimate() -> float:
    """Estimate hydrogen's ground-state energy by minimizing E(r) = hbar^2/(2 m r^2) -
    e^2/(4 pi eps0 r) over the size r. Returns the binding energy magnitude in joules;
    comes out ~13.6 eV -- the uncertainty principle alone sets the atomic scale."""
    eps0 = 8.8541878128e-12
    e = 1.602176634e-19
    k = e * e / (4.0 * math.pi * eps0)
    # minimize hbar^2/(2 m r^2) - k/r  ->  r_min = hbar^2/(m k), E_min = -m k^2/(2 hbar^2)
    return M_E * k * k / (2.0 * HBAR * HBAR)

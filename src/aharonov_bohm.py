"""The Aharonov-Bohm effect: a phase shift from a field you never touch.

Classically only forces matter, and where there is no field there is no force. Quantum
mechanics disagrees. Send an electron beam around both sides of a thin solenoid -- with the
magnetic field entirely confined inside, zero on the electron's path -- and the interference
pattern still shifts. The electron responds to the vector potential A, not the field B, and
picks up a phase

    delta_phi = (q / hbar) * closed integral of A . dl = q Phi / hbar,

set purely by the enclosed magnetic flux Phi, even though B = 0 everywhere the electron went.
Aharonov and Bohm predicted it in 1959; it is one of the sharpest demonstrations that the
potentials are physically real in quantum mechanics, not just mathematical bookkeeping.

The phase is periodic in the flux with period the magnetic flux quantum

    Phi_0 = h / q  ( = 4.14e-15 Wb for an electron ),

so the fringe pattern repeats every time one more flux quantum threads the loop. In
superconductors the relevant charge is the Cooper pair 2e, giving the smaller
superconducting flux quantum h/2e = 2.07e-15 Wb, which quantizes flux through a ring and runs
SQUID magnetometers -- the most sensitive field detectors made.

This module gives the AB phase shift, the flux quantum (single-charge and superconducting),
the number of flux quanta in a flux, the fringe shift in units of a period, and the flux from
a field through a loop, and reproduces the h/q period and the SQUID flux quantum. SI units.
Pure stdlib; the quantum-phase companion to the uncertainty and de Broglie notes.
"""

from __future__ import annotations

import math

E_CHARGE = 1.602176634e-19
HBAR = 1.054571817e-34
H = 6.62607015e-34

FLUX_QUANTUM = H / E_CHARGE               # single-electron flux quantum h/e (Wb)
FLUX_QUANTUM_SC = H / (2.0 * E_CHARGE)    # superconducting flux quantum h/2e (Wb)


def phase_shift(flux: float, charge: float = E_CHARGE) -> float:
    """Aharonov-Bohm phase shift delta_phi = q Phi / hbar (rad) from enclosed magnetic flux
    Phi, independent of the field along the path (which can be zero)."""
    return charge * flux / HBAR


def flux_quantum(charge: float = E_CHARGE) -> float:
    """Magnetic flux quantum Phi_0 = h / q (Wb): the flux that advances the AB phase by 2 pi.
    4.14e-15 Wb for a single electron, 2.07e-15 for a Cooper pair (2e)."""
    return H / charge


def num_flux_quanta(flux: float, charge: float = E_CHARGE) -> float:
    """Number of flux quanta threading the loop, Phi / Phi_0 = q Phi / h. The AB phase is
    2 pi times this; the interference pattern repeats each integer."""
    return charge * flux / H


def fringe_shift(flux: float, charge: float = E_CHARGE) -> float:
    """Interference-fringe shift in units of one full period (fractional), = the AB phase
    modulo 2 pi expressed as a fraction. 0 means the pattern is back to unshifted."""
    return num_flux_quanta(flux, charge) % 1.0


def flux_through_loop(field: float, area: float) -> float:
    """Magnetic flux Phi = B * A (Wb) through a loop of area A in a uniform field B. (For the
    AB effect the field can be confined off the path; this is the enclosed flux either way.)"""
    return field * area


def field_for_one_quantum(area: float, charge: float = E_CHARGE) -> float:
    """Uniform field (T) that puts exactly one flux quantum through a loop of area A:
    B = Phi_0 / A. Tiny for a large loop -- why SQUIDs sense minute fields."""
    return flux_quantum(charge) / area

"""The Josephson junction: a supercurrent that tunnels, and defines the volt.

Put a thin insulating barrier between two superconductors and Cooper pairs tunnel straight
through it with no voltage at all -- a dissipationless supercurrent set only by the quantum
phase difference phi across the junction. Brian Josephson predicted it in 1962 (as a graduate
student); it earned him the Nobel Prize. Two relations govern it.

DC Josephson effect: a zero-voltage current

    I = I_c sin(phi),

flows for any phase phi, up to a maximum critical current I_c. AC Josephson effect: hold a
DC voltage V across the junction and the phase winds at

    d(phi)/dt = 2 e V / hbar,

so the supercurrent oscillates at the Josephson frequency

    f = 2 e V / h = V / Phi_0  = 483.6 GHz per millivolt,

an exact voltage-to-frequency conversion through only fundamental constants. Because frequency
can be measured to extraordinary precision, this makes the junction the modern definition of
the volt: irradiate it at frequency f and it locks onto quantized voltage steps (Shapiro
steps) at V = n h f / 2e.

This module gives the DC supercurrent, the Josephson frequency and its inverse, the Josephson
constant K_J = 2e/h, the Shapiro-step voltages, and the junction coupling energy, and
reproduces the 483.6 GHz/mV conversion and the sin(phi) current. SI units. Pure stdlib; the
superconducting-phase companion to the Aharonov-Bohm and BCS-adjacent notes.
"""

from __future__ import annotations

import math

E_CHARGE = 1.602176634e-19
HBAR = 1.054571817e-34
H = 6.62607015e-34

JOSEPHSON_CONSTANT = 2.0 * E_CHARGE / H   # K_J = 2e/h ~ 4.836e14 Hz/V
FLUX_QUANTUM_SC = H / (2.0 * E_CHARGE)    # h/2e ~ 2.068e-15 Wb


def supercurrent(critical_current: float, phase: float) -> float:
    """DC Josephson current I = I_c sin(phi) (A): a zero-voltage supercurrent set by the phase
    difference phi, up to the critical current I_c."""
    return critical_current * math.sin(phase)


def josephson_frequency(voltage: float) -> float:
    """Josephson (AC) frequency f = 2 e V / h (Hz) at which the supercurrent oscillates under
    a DC voltage V. 483.6 GHz per millivolt -- exact, from fundamental constants."""
    return JOSEPHSON_CONSTANT * voltage


def voltage_from_frequency(frequency: float) -> float:
    """Voltage V = h f / (2 e) (V) that produces a given Josephson frequency -- the volt
    standard: measure f, get V exactly. Inverts josephson_frequency."""
    return frequency / JOSEPHSON_CONSTANT


def josephson_constant() -> float:
    """The Josephson constant K_J = 2e/h ~ 4.836e14 Hz/V, the exact volt-to-frequency ratio
    that defines the SI volt."""
    return JOSEPHSON_CONSTANT


def shapiro_step_voltage(n: int, frequency: float) -> float:
    """Voltage V_n = n h f / (2e) (V) of the n-th Shapiro step: quantized voltage plateaus
    that appear when the junction is irradiated at frequency f. The basis of voltage metrology."""
    return n * frequency / JOSEPHSON_CONSTANT


def phase_evolution_rate(voltage: float) -> float:
    """Rate of change of the junction phase, d(phi)/dt = 2 e V / hbar (rad/s), under a DC
    voltage V -- the phase winds, driving the AC supercurrent."""
    return 2.0 * E_CHARGE * voltage / HBAR


def coupling_energy(critical_current: float) -> float:
    """Josephson coupling energy E_J = hbar I_c / (2e) (J): the energy scale of the phase
    coupling across the junction, which sets its behaviour as a qubit or oscillator."""
    return HBAR * critical_current / (2.0 * E_CHARGE)

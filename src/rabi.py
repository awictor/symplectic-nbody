"""Rabi oscillations: a two-level atom flopping in a driving field.

Shine a near-resonant field on a two-level system -- an atom, a spin, a qubit -- and it does
not just absorb once and stop. It cycles coherently between the ground and excited states,
Rabi flopping. On exact resonance the probability of finding it excited is

    P_e(t) = sin^2(Omega t / 2),

oscillating fully between 0 and 1 at the Rabi frequency Omega = d E / hbar (the coupling of
the transition dipole d to the field amplitude E). A pulse that leaves Omega t = pi -- a "pi
pulse" -- fully inverts the population (an X gate on a qubit); a pi/2 pulse makes an equal
superposition.

Off resonance, by a detuning delta = omega_drive - omega_0, the atom still oscillates but at
the faster generalized Rabi frequency

    Omega_R = sqrt(Omega^2 + delta^2),

and it never fully reaches the excited state -- the population only climbs to

    P_max = Omega^2 / (Omega^2 + delta^2),

a Lorentzian in detuning of width Omega. That resonance curve is how you find and calibrate a
transition, and its linewidth (power broadening) grows with drive strength.

This module gives the on- and off-resonant excitation probability, the Rabi and generalized
Rabi frequencies, the pi- and pi/2-pulse durations, and the resonance-peak height versus
detuning, and reproduces the full-contrast on-resonance flopping and the Lorentzian
line shape. SI-consistent (Omega and delta in rad/s, t in s). Pure stdlib; the quantum-
dynamics companion to the two-level Bohr and uncertainty notes.
"""

from __future__ import annotations

import math

HBAR = 1.054571817e-34        # J s


def rabi_frequency(dipole: float, field_amplitude: float) -> float:
    """Rabi frequency Omega = d E / hbar (rad/s): the coupling of a transition dipole moment d
    to a driving field amplitude E. Sets how fast the atom flops."""
    return dipole * field_amplitude / HBAR


def generalized_rabi(omega: float, detuning: float) -> float:
    """Generalized Rabi frequency Omega_R = sqrt(Omega^2 + delta^2) (rad/s) for a drive
    detuned by delta from resonance. Faster than the bare Omega, and never fully inverts."""
    return math.sqrt(omega * omega + detuning * detuning)


def excited_probability(omega: float, detuning: float, t: float) -> float:
    """Probability of being in the excited state at time t:
    P_e = (Omega^2 / Omega_R^2) sin^2(Omega_R t / 2). On resonance (delta=0) this is
    sin^2(Omega t/2), swinging fully between 0 and 1."""
    omega_r = generalized_rabi(omega, detuning)
    if omega_r == 0.0:
        return 0.0
    return (omega * omega / (omega_r * omega_r)) * math.sin(omega_r * t / 2.0) ** 2


def peak_probability(omega: float, detuning: float) -> float:
    """Maximum excited-state probability over a cycle, Omega^2 / (Omega^2 + delta^2): a
    Lorentzian in detuning of width Omega. 1 on resonance, falling off the peak."""
    return omega * omega / (omega * omega + detuning * detuning)


def pi_pulse_time(omega: float) -> float:
    """Duration of a pi pulse on resonance, t = pi / Omega (s): fully inverts the population
    (ground -> excited), the qubit X gate."""
    return math.pi / omega


def half_pi_pulse_time(omega: float) -> float:
    """Duration of a pi/2 pulse, t = pi / (2 Omega) (s): drives an equal ground-excited
    superposition (a Hadamard-like rotation)."""
    return math.pi / (2.0 * omega)

"""Mean-field Ising ferromagnetism: how order appears from disorder.

Below a critical temperature a magnet spontaneously magnetizes: countless spins, each favouring
alignment with its neighbours, tip collectively into one direction with no external field. The
simplest theory that captures it is Weiss mean-field theory of the Ising model. Each spin feels
an effective field made of the applied field plus the average alignment of its z neighbours, so
the self-consistent magnetization per spin obeys

    m = tanh( (z J m + mu_B B) / (k_B T) ),

with z the coordination number, J the coupling, and m in [-1, 1]. Above the Curie temperature

    T_c = z J / k_B,

the only solution at zero field is m = 0 (paramagnet, disordered). Below it a nonzero m appears
spontaneously -- the ferromagnetic phase. Near T_c the magnetization vanishes as a power law

    m ~ (1 - T/T_c)^(1/2),

the mean-field critical exponent beta = 1/2 (the true 3D value is ~0.33; mean field ignores
fluctuations but gets the qualitative transition right). The zero-field susceptibility diverges
at T_c as the Curie-Weiss law chi ~ 1 / (T - T_c), the divergence that signals a second-order
phase transition.

This module gives the Curie temperature, the self-consistent spontaneous magnetization (by
iteration), the paramagnetic/ferromagnetic classification, the near-T_c critical scaling, and
the Curie-Weiss susceptibility, and reproduces the m->1 low-T limit, the m=0 paramagnet above
T_c, and the beta=1/2 exponent. Dimensionless (energies in units of k_B). Pure stdlib; the
phase-transition companion to the BCS and thermodynamics notes.
"""

from __future__ import annotations

import math


def curie_temperature(coupling: float, coordination: int) -> float:
    """Mean-field Curie temperature T_c = z J / k_B (here in energy units, so T_c = z J with
    k_B = 1). Above it the magnet is disordered; below it, spontaneously magnetized."""
    return coordination * coupling


def spontaneous_magnetization(temperature: float, coupling: float, coordination: int,
                              field: float = 0.0, iterations: int = 200) -> float:
    """Self-consistent magnetization m = tanh((z J m + B)/T), solved by iteration from a
    fully-aligned start. Returns |m| in [0,1]: 0 above T_c at zero field, rising toward 1 as
    T -> 0. (Energies in k_B units.)"""
    if temperature <= 0.0:
        return 1.0
    tc = curie_temperature(coupling, coordination)
    m = 1.0 if (temperature < tc or field != 0.0) else 0.01   # seed
    for _ in range(iterations):
        m_new = math.tanh((coordination * coupling * m + field) / temperature)
        if abs(m_new - m) < 1e-12:
            m = m_new
            break
        m = m_new
    return abs(m)


def is_ferromagnetic(temperature: float, coupling: float, coordination: int) -> bool:
    """True if the system is below the Curie temperature (ordered, spontaneously magnetized)
    at zero field."""
    return temperature < curie_temperature(coupling, coordination)


def critical_magnetization(temperature: float, tc: float) -> float:
    """Near-T_c mean-field magnetization m ~ (1 - T/T_c)^(1/2) (the beta=1/2 exponent).
    Zero at and above T_c. A closed-form approximation to the self-consistent solution."""
    if temperature >= tc:
        return 0.0
    return math.sqrt(1.0 - temperature / tc)


def curie_weiss_susceptibility(temperature: float, tc: float, curie_constant: float = 1.0) -> float:
    """Zero-field susceptibility chi = C / (T - T_c) (Curie-Weiss law) in the paramagnetic
    phase T > T_c. Diverges as T -> T_c^+, the signature of the transition."""
    if temperature <= tc:
        return float("inf")
    return curie_constant / (temperature - tc)


def reduced_temperature(temperature: float, tc: float) -> float:
    """Reduced temperature t = (T - T_c) / T_c: the scaling variable near the transition,
    negative in the ordered phase, positive in the disordered one."""
    return (temperature - tc) / tc

"""Langevin paramagnetism: magnetic moments lining up against thermal chaos.

A paramagnet is a crowd of permanent magnetic moments -- each free to point any way -- with no
coupling between them (unlike a ferromagnet). Apply a field B and each moment mu feels an
energy -mu.B that favours alignment; temperature fights back, randomizing directions. The
balance is pure Boltzmann statistics, and averaging cos(theta) over the thermal distribution
gives the classical result Langevin found in 1905: the magnetization per moment is

    m / mu = L(x) = coth(x) - 1/x,        x = mu B / (k_B T),

the Langevin function. At small x (weak field or high temperature) it is linear, L(x) ~ x/3,
so the susceptibility follows Curie's law

    chi ~ n mu^2 / (3 k_B T)  ~  1/T,

the inverse-temperature falloff that defines a paramagnet. At large x it saturates, L -> 1:
every moment is aligned and the magnetization cannot grow further. Quantum mechanically the
continuous L(x) is replaced by the Brillouin function for discrete spin states, but for large
J or classical moments they agree.

This module gives the Langevin function, the reduced field x, the magnetization and its
saturation, the small-field Curie susceptibility, and the field or temperature for a target
alignment, and reproduces the L(x) -> x/3 Curie limit and the L -> 1 saturation. SI units.
Pure stdlib; the statistical-magnetism companion to the Ising and thermodynamics notes.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
BOHR_MAGNETON = 9.2740100783e-24    # J/T


def reduced_field(moment: float, field: float, temperature: float) -> float:
    """Reduced field x = mu B / (k_B T): the ratio of magnetic alignment energy to thermal
    energy. Small x is the linear Curie regime, large x saturates."""
    return moment * field / (K_B * temperature)


def langevin(x: float) -> float:
    """Langevin function L(x) = coth(x) - 1/x, the thermal average of cos(theta). Runs from 0
    (x=0) to 1 (x -> inf); L(x) ~ x/3 for small x. Series-expanded near 0 for stability."""
    if abs(x) < 1e-4:
        return x / 3.0 - x ** 3 / 45.0
    return 1.0 / math.tanh(x) - 1.0 / x


def magnetization(moment: float, field: float, temperature: float, n: float = 1.0) -> float:
    """Magnetization n mu L(x) (A/m if n is number density) for n moments of size mu at field
    B and temperature T. Per-moment if n=1."""
    return n * moment * langevin(reduced_field(moment, field, temperature))


def saturation_fraction(moment: float, field: float, temperature: float) -> float:
    """Fraction of full alignment, m / (n mu) = L(x), between 0 and 1. Near 1 means the
    moments are essentially all aligned (high field / low temperature)."""
    return langevin(reduced_field(moment, field, temperature))


def curie_susceptibility(moment: float, temperature: float, n: float = 1.0) -> float:
    """Small-field (Curie-law) susceptibility chi = n mu^2 / (3 k_B T): the linear response
    m = chi B, falling as 1/T. The signature of a paramagnet."""
    return n * moment * moment / (3.0 * K_B * temperature)


def curie_constant(moment: float, n: float = 1.0) -> float:
    """Curie constant C = n mu^2 / (3 k_B), so chi = C / T. Material property independent of
    temperature."""
    return n * moment * moment / (3.0 * K_B)


def field_for_saturation(moment: float, temperature: float, fraction: float,
                         hi: float = 1e4) -> float:
    """Field (T) needed to reach a target alignment fraction L(x)=fraction at temperature T,
    by bisection on x. Grows without bound as fraction -> 1 (full saturation is asymptotic)."""
    if fraction <= 0.0:
        return 0.0
    lo, hi_x = 0.0, hi
    for _ in range(200):
        mid = 0.5 * (lo + hi_x)
        if langevin(mid) < fraction:
            lo = mid
        else:
            hi_x = mid
    x = 0.5 * (lo + hi_x)
    return x * K_B * temperature / moment

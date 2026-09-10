"""The Maxwell-Boltzmann speed distribution: how fast gas molecules move.

In a gas at temperature T the molecular speeds follow the Maxwell-Boltzmann
distribution,

    f(v) = 4 pi (m / 2 pi k_B T)^(3/2) v^2 exp(-m v^2 / 2 k_B T),

the product of a rising phase-space factor v^2 and a falling Boltzmann factor. Three
characteristic speeds fall out of it, always in the same ratio:

    most probable   v_p   = sqrt(2 k_B T / m)              (peak of f)
    mean            <v>   = sqrt(8 k_B T / pi m) = 1.128 v_p
    root-mean-square v_rms = sqrt(3 k_B T / m)   = 1.225 v_p

so v_p : <v> : v_rms = 1 : 1.128 : 1.225, independent of gas or temperature. The rms
speed is what sets the pressure and the (3/2) k_B T of kinetic energy; the mean speed
sets collision and effusion rates.

The high-speed tail decays as exp(-v^2), which is why only an exponentially small
fraction of molecules ever exceed several times v_p -- the same tail that governs
thermal (Jeans) atmospheric escape and the onset of thermonuclear reactions. Nitrogen
in room air has v_rms ~ 500 m/s (faster than sound); hydrogen at the same temperature
is ~4x faster because v ~ 1/sqrt(m).

This module gives the distribution, the three characteristic speeds, the mean kinetic
energy, and the fraction of molecules above a threshold speed, and reproduces the
1 : 1.128 : 1.225 ratio and room-temperature air speeds. SI units. Pure stdlib; the
kinetic-theory companion to the Jeans-escape and Sackur-Tetrode modules.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
AMU = 1.66053907e-27


def distribution(v: float, T: float, m: float) -> float:
    """Maxwell-Boltzmann speed probability density f(v) (s/m):
    4 pi (m/2 pi kT)^(3/2) v^2 exp(-m v^2 / 2 kT)."""
    a = m / (2.0 * math.pi * K_B * T)
    return 4.0 * math.pi * a ** 1.5 * v * v * math.exp(-m * v * v / (2.0 * K_B * T))


def most_probable_speed(T: float, m: float) -> float:
    """Most probable speed v_p = sqrt(2 k_B T / m) (m/s): the peak of f(v)."""
    return math.sqrt(2.0 * K_B * T / m)


def mean_speed(T: float, m: float) -> float:
    """Mean speed <v> = sqrt(8 k_B T / pi m) = 1.128 v_p (m/s)."""
    return math.sqrt(8.0 * K_B * T / (math.pi * m))


def rms_speed(T: float, m: float) -> float:
    """Root-mean-square speed v_rms = sqrt(3 k_B T / m) = 1.225 v_p (m/s)."""
    return math.sqrt(3.0 * K_B * T / m)


def mean_kinetic_energy(T: float) -> float:
    """Average translational kinetic energy (3/2) k_B T (J), independent of mass."""
    return 1.5 * K_B * T


def fraction_above(v_threshold: float, T: float, m: float,
                   steps: int = 20000) -> float:
    """Fraction of molecules with speed above v_threshold, by numerically integrating
    f(v) from the threshold out into the tail (trapezoid rule)."""
    vp = most_probable_speed(T, m)
    v_max = v_threshold + 12.0 * vp        # the exp(-v^2) tail is negligible past this
    if v_max <= v_threshold:
        return 0.0
    dv = (v_max - v_threshold) / steps
    total = 0.0
    prev = distribution(v_threshold, T, m)
    for i in range(1, steps + 1):
        v = v_threshold + i * dv
        cur = distribution(v, T, m)
        total += 0.5 * (prev + cur) * dv
        prev = cur
    return total

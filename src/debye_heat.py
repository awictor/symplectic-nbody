"""The Debye model: why solids stop storing heat at low temperature.

Classically each atom in a solid holds 3 k_B T of vibrational energy, so the molar heat
capacity should be a constant 3R ~ 24.9 J/(mol K) -- the Dulong-Petit law. It works at
room temperature but fails badly in the cold: measured heat capacities plunge toward
zero as T -> 0, which classical physics cannot explain.

Debye treated the solid's vibrations as a gas of quantized sound waves (phonons) with a
maximum frequency set by the interatomic spacing, giving a characteristic Debye
temperature Theta_D. The molar heat capacity is then

    C_V = 9 R (T/Theta_D)^3 integral_0^{Theta_D/T} x^4 e^x / (e^x - 1)^2 dx,

which has two clean limits:

    T >> Theta_D:  C_V -> 3R          (Dulong-Petit recovered)
    T << Theta_D:  C_V -> (12 pi^4 / 5) R (T/Theta_D)^3   (the Debye T^3 law).

The T^3 falloff is a direct consequence of phonon quantization -- freezing out
high-frequency modes -- and it matches experiment across materials. Theta_D is high for
stiff, light lattices (diamond ~2230 K, so diamond is still "cold" at room temperature)
and low for soft, heavy ones (lead ~105 K).

This module gives the Debye heat capacity by numerical integration, its high- and
low-temperature limits, and the Dulong-Petit value, and reproduces the T^3 law and the
3R plateau. SI units (J/mol/K). Pure stdlib; the lattice-heat companion to the
quantum-statistics and Sackur-Tetrode modules.
"""

from __future__ import annotations

import math

R_GAS = 8.314462618            # molar gas constant (J/mol/K)
DULONG_PETIT = 3.0 * R_GAS     # classical molar heat capacity ~ 24.94 J/mol/K

# representative Debye temperatures (K)
THETA_DIAMOND = 2230.0
THETA_COPPER = 343.0
THETA_LEAD = 105.0


def _debye_integrand(x: float) -> float:
    """x^4 e^x / (e^x - 1)^2, guarded near x=0 (limit x^2) and large x."""
    if x < 1e-6:
        return x * x           # small-x limit
    if x > 700.0:
        return 0.0
    ex = math.exp(x)
    return x ** 4 * ex / (ex - 1.0) ** 2


def heat_capacity(T: float, theta_D: float, steps: int = 2000) -> float:
    """Debye molar heat capacity C_V (J/mol/K) at temperature T and Debye temperature
    theta_D, by trapezoid integration of the Debye function."""
    if T <= 0.0:
        return 0.0
    upper = theta_D / T
    dx = upper / steps
    total = 0.0
    prev = _debye_integrand(0.0)
    for i in range(1, steps + 1):
        cur = _debye_integrand(i * dx)
        total += 0.5 * (prev + cur) * dx
        prev = cur
    return 9.0 * R_GAS * (T / theta_D) ** 3 * total


def low_temperature_limit(T: float, theta_D: float) -> float:
    """Low-T Debye C_V = (12 pi^4 / 5) R (T/theta_D)^3 (J/mol/K): the T^3 law."""
    return 12.0 * math.pi ** 4 / 5.0 * R_GAS * (T / theta_D) ** 3


def high_temperature_limit() -> float:
    """High-T limit: the Dulong-Petit value 3R (J/mol/K)."""
    return DULONG_PETIT


def fraction_of_dulong_petit(T: float, theta_D: float) -> float:
    """Fraction of the classical 3R that the Debye C_V reaches at temperature T."""
    return heat_capacity(T, theta_D) / DULONG_PETIT

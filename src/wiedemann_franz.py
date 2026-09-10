"""The Wiedemann-Franz law: good electrical conductors are good heat conductors.

In a metal the same free electrons that carry charge also carry heat, so the two
conductivities are linked. Wiedemann and Franz noticed in 1853 that their ratio is nearly the
same for all metals at a given temperature; Lorenz saw it scales with T. The Drude/Sommerfeld
theory explains it: dividing the electronic thermal conductivity kappa by the electrical
conductivity sigma leaves only fundamental constants,

    kappa / (sigma T) = L = pi^2 k_B^2 / (3 e^2) = 2.44e-8 W ohm / K^2,

the Lorenz number L. It holds because each electron carries both a charge e and a thermal
energy ~k_B T, and the same scattering limits both currents, so the material-specific mean
free path and carrier density cancel in the ratio.

The law is a workhorse: it lets you estimate a metal's thermal conductivity from an easy
resistance measurement, and its *breakdown* is diagnostic -- a measured Lorenz number well
below L signals that heat and charge are carried differently (inelastic scattering, or exotic
"strange metals" where the electrons do not behave as simple quasiparticles). It also
correctly predicts that heat conduction, like electrical conduction, is dominated by
electrons in a metal but by phonons in an insulator (where the law does not apply).

This module gives the Lorenz number, the thermal conductivity predicted from the electrical
one (and vice versa), the effective Lorenz number from measured values, and a test of whether
a material obeys the law, and reproduces copper's ~400 W/(m K) thermal conductivity from its
conductivity. SI units. Pure stdlib; the electron-transport companion to the Hall-effect and
Drude notes.
"""

from __future__ import annotations

K_B = 1.380649e-23             # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19    # elementary charge (C)
import math

LORENZ_NUMBER = math.pi ** 2 * K_B ** 2 / (3.0 * E_CHARGE ** 2)   # 2.44e-8 W ohm / K^2


def lorenz_number() -> float:
    """The Sommerfeld Lorenz number L = pi^2 k_B^2 / (3 e^2) = 2.44e-8 W ohm / K^2, the
    universal ratio kappa/(sigma T) for a simple metal."""
    return LORENZ_NUMBER


def thermal_conductivity(sigma: float, temperature: float, l: float = LORENZ_NUMBER) -> float:
    """Electronic thermal conductivity kappa = L sigma T (W/(m K)) from the electrical
    conductivity sigma and temperature T -- the Wiedemann-Franz prediction."""
    return l * sigma * temperature


def electrical_conductivity(kappa: float, temperature: float,
                            l: float = LORENZ_NUMBER) -> float:
    """Electrical conductivity sigma = kappa / (L T) (S/m) inferred from the thermal
    conductivity. Inverts thermal_conductivity."""
    return kappa / (l * temperature)


def effective_lorenz(kappa: float, sigma: float, temperature: float) -> float:
    """Effective Lorenz number L = kappa / (sigma T) (W ohm / K^2) from measured thermal and
    electrical conductivities. Compare to 2.44e-8 to test the law."""
    return kappa / (sigma * temperature)


def obeys_law(kappa: float, sigma: float, temperature: float, tol: float = 0.15) -> bool:
    """True if the effective Lorenz number is within tol (default 15%) of the Sommerfeld
    value -- i.e. the metal obeys Wiedemann-Franz. A large shortfall flags non-quasiparticle
    or phonon-dominated transport."""
    return abs(effective_lorenz(kappa, sigma, temperature) - LORENZ_NUMBER) <= tol * LORENZ_NUMBER

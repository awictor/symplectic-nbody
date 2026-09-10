"""The Saha equation and cosmic recombination: when the universe went neutral.

In thermal equilibrium the ionization fraction of a gas is set by the Saha
equation, balancing ionization against recombination:

    (1 - x) / x^2 = n_b (2 pi m_e k T / h^2)^{-3/2} exp(chi / kT),

where x = n_e/n_H is the ionized fraction, n_b the baryon (proton) number
density, and chi = 13.6 eV the hydrogen binding energy.

The striking result: hydrogen does not recombine at kT ~ chi (which would be
~158000 K). Because there are ~1.6 billion photons per baryon, the rare
high-energy photons in the tail keep hydrogen ionized until the temperature drops
to ~3700 K -- redshift z ~ 1100, ~380000 years after the Big Bang. That is when
the universe became transparent and released the cosmic microwave background.

This module solves the Saha equation for the ionization fraction as a function of
temperature and redshift, and finds the recombination redshift. SI units. Pure
stdlib.
"""

from __future__ import annotations

import math

K_B = 1.380649e-23
H_PLANCK = 6.62607015e-34
M_E = 9.1093837e-31
EV = 1.602176634e-19
CHI_H = 13.6 * EV               # hydrogen ionization energy, J

# cosmological baryon density today: eta ~ 6.1e-10 photons... actually n_b/n_gamma
ETA = 6.1e-10                   # baryon-to-photon ratio
# CMB photon number density today n_gamma0 ~ 4.11e8 /m^3 (T0=2.725 K)
N_GAMMA0 = 4.11e8
T0 = 2.725                      # CMB temperature today, K


def baryon_density(z: float) -> float:
    """Baryon number density at redshift z: n_b = eta n_gamma0 (1+z)^3."""
    return ETA * N_GAMMA0 * (1.0 + z) ** 3


def saha_ionization_fraction(T: float, n_b: float) -> float:
    """Solve the Saha equation (1-x)/x^2 = S for the ionized fraction x in [0,1],
    with S = n_b (2 pi m_e k T / h^2)^{-3/2} exp(chi/kT)."""
    thermal = (2.0 * math.pi * M_E * K_B * T / H_PLANCK ** 2) ** 1.5
    S = n_b / thermal * math.exp(CHI_H / (K_B * T))
    # x^2 S + x - 1 = 0  ->  x = (-1 + sqrt(1 + 4 S)) / (2 S)
    if S <= 0.0:
        return 1.0
    return (-1.0 + math.sqrt(1.0 + 4.0 * S)) / (2.0 * S)


def ionization_at_redshift(z: float) -> float:
    """Ionization fraction at redshift z, with T = T0 (1+z) and the cosmic
    baryon density."""
    T = T0 * (1.0 + z)
    return saha_ionization_fraction(T, baryon_density(z))


def recombination_redshift(x_target: float = 0.5) -> float:
    """Redshift at which the ionization fraction falls to x_target (default 0.5,
    the conventional recombination point ~1100). Bisection in z."""
    # ionization DECREASES with decreasing z (cooler). Bracket the crossing.
    lo, hi = 1000.0, 1800.0     # lo: low z (less ionized), hi: high z (more ionized)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if ionization_at_redshift(mid) > x_target:
            hi = mid            # too ionized -> need lower z (cooler)
        else:
            lo = mid            # too neutral -> need higher z (hotter)
    return 0.5 * (lo + hi)


def naive_ionization_temperature() -> float:
    """The naive (wrong) estimate kT = chi -> T ~ 158000 K, ignoring the huge
    photon-to-baryon ratio."""
    return CHI_H / K_B

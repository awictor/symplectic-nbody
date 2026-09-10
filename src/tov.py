"""Neutron-star structure: the Tolman-Oppenheimer-Volkoff (TOV) equation.

A neutron star is held up by neutron degeneracy pressure against its own crushing
gravity, but now the gravity is STRONG enough that Newtonian hydrostatic
equilibrium is wrong -- you need general relativity. The relativistic equation of
hydrostatic equilibrium is the TOV equation (geometrized units G = c = 1):

    dP/dr = - (rho + P)(m + 4 pi r^3 P) / ( r (r - 2 m) )
    dm/dr = 4 pi r^2 rho

The extra factors over the Newtonian -G m rho / r^2 are all relativistic
corrections; each makes gravity effectively stronger, so -- unlike the Newtonian
case -- there is a MAXIMUM mass beyond which no static star exists. For the
original Oppenheimer-Volkoff ideal degenerate neutron gas that limit is ~0.7
solar masses (real EOS with nuclear interactions push it to ~2 M_sun).

This module integrates the TOV equation for a polytropic EOS, traces the
mass-radius relation, and finds the maximum mass. It also compares TOV to the
Newtonian structure to show GR lowers the maximum supportable mass. Distances in
km, masses in solar masses (via geometrized conversions). Pure stdlib.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

# geometrized: 1 solar mass = 1.4766 km (G Msun / c^2); densities in 1/km^2
MSUN_KM = 1.476625


def tov_rhs(r: float, P: float, m: float, rho_of_P: Callable[[float], float],
            newtonian: bool = False):
    """Return (dP/dr, dm/dr). If newtonian=True, drop the GR correction factors
    (recovering dP/dr = -m rho / r^2) to contrast the two."""
    rho = rho_of_P(P)
    dm = 4.0 * math.pi * r * r * rho
    if r < 1e-12:
        return 0.0, dm
    if newtonian:
        dP = -m * rho / (r * r)
    else:
        dP = -(rho + P) * (m + 4.0 * math.pi * r ** 3 * P) / (r * (r - 2.0 * m))
    return dP, dm


def integrate_star(rho_c: float, K: float, gamma: float,
                   dr: float = 1e-3, newtonian: bool = False
                   ) -> Tuple[float, float]:
    """Integrate one star with a polytropic EOS P = K rho^gamma from central
    density rho_c outward until P -> 0 (surface). Returns (radius_km, mass_Msun).
    Units: geometrized (lengths in km, so densities in km^-2)."""
    def P_of_rho(rho):
        return K * rho ** gamma

    def rho_of_P(P):
        return (P / K) ** (1.0 / gamma) if P > 0 else 0.0

    r = dr
    rho = rho_c
    P = P_of_rho(rho_c)
    m = 4.0 / 3.0 * math.pi * r ** 3 * rho_c
    P_surface = 1e-14 * P
    while P > P_surface:
        # RK4 on (P, m)
        def f(r, P, m):
            return tov_rhs(r, P, m, rho_of_P, newtonian)
        k1P, k1m = f(r, P, m)
        k2P, k2m = f(r + 0.5 * dr, P + 0.5 * dr * k1P, m + 0.5 * dr * k1m)
        k3P, k3m = f(r + 0.5 * dr, P + 0.5 * dr * k2P, m + 0.5 * dr * k2m)
        k4P, k4m = f(r + dr, P + dr * k3P, m + dr * k3m)
        P_new = P + dr / 6.0 * (k1P + 2 * k2P + 2 * k3P + k4P)
        m += dr / 6.0 * (k1m + 2 * k2m + 2 * k3m + k4m)
        r += dr
        if P_new <= 0.0 or (not newtonian and r <= 2.0 * m):
            break
        P = P_new
    return r, m / MSUN_KM


def mass_radius_sequence(rho_cs: List[float], K: float, gamma: float,
                         newtonian: bool = False):
    """Return [(R_km, M_Msun), ...] for a list of central densities."""
    return [integrate_star(rc, K, gamma, newtonian=newtonian) for rc in rho_cs]


def maximum_mass(seq: List[Tuple[float, float]]) -> float:
    """Maximum mass (Msun) in a mass-radius sequence."""
    return max(m for _r, m in seq)

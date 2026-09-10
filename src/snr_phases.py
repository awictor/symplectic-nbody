"""Supernova-remnant evolution: the four ages of a blast wave.

A supernova dumps ~10^51 erg of kinetic energy into the interstellar medium, and the
resulting shell evolves through four distinct phases as it sweeps up mass and radiates
energy away.

1. FREE EXPANSION. The ejecta coast ballistically at their initial velocity, R ~ v t,
   until they have swept up their own mass. This ends at the sweep-up radius

       R_sweep = (3 M_ej / (4 pi rho))^(1/3),

   typically a few parsecs after a few hundred years.

2. SEDOV-TAYLOR (adiabatic). Swept-up mass now dominates but radiative losses are
   still negligible, so energy is conserved and the shock is self-similar,

       R ~ (E t^2 / rho)^(1/5),   R ~ t^(2/5).

   This is the long-lived phase (tens of thousands of years) covered in detail by the
   sedov module.

3. SNOWPLOW (radiative / pressure-driven). Once the post-shock gas cools below ~10^6 K
   it radiates efficiently, forming a dense cool shell that coasts on its momentum
   while the hot interior pressure still pushes: R ~ t^(2/7). The transition happens at
   the cooling time, at a radius of ~20-30 pc after ~few x 10^4 yr.

4. MERGE / DISPERSAL. When the expansion speed drops to the ISM sound speed / turbulent
   velocity (~10 km/s), the shell can no longer be distinguished from the surrounding
   medium and merges into it, after ~10^6 yr and ~50-100 pc.

This module gives the sweep-up radius ending free expansion, the Sedov radius and
speed, the radius/time scalings of each phase, and the merge radius, and reproduces the
canonical few-pc / tens-of-pc / ~100-pc progression. SI units. Pure stdlib; the
life-cycle companion to the sedov and shock-jump modules.
"""

from __future__ import annotations

import math

PC = 3.0856775814913673e16
M_SUN = 1.989e30
M_P = 1.6726219e-27
YEAR = 3.15576e7
KM = 1e3
E_SN = 1e44                    # canonical supernova kinetic energy, 10^51 erg = 1e44 J

XI_SEDOV = 1.15               # Sedov similarity constant (gamma=5/3)


def ism_density(n_per_cc: float, mu: float = 1.4) -> float:
    """ISM mass density (kg/m^3) from hydrogen number density (per cc)."""
    return mu * M_P * n_per_cc * 1e6


def sweep_up_radius(M_ej: float, rho: float) -> float:
    """Radius (m) where swept-up ISM mass equals the ejecta mass, ending free
    expansion: R = (3 M_ej / (4 pi rho))^(1/3)."""
    return (3.0 * M_ej / (4.0 * math.pi * rho)) ** (1.0 / 3.0)


def free_expansion_end_time(M_ej: float, rho: float, v_ej: float) -> float:
    """Approximate time (s) free expansion ends: R_sweep / v_ej."""
    return sweep_up_radius(M_ej, rho) / v_ej


def sedov_radius(t: float, E: float = E_SN, rho: float = None,
                 n_per_cc: float = 1.0, xi: float = XI_SEDOV) -> float:
    """Sedov-Taylor blast radius (m): R = xi (E t^2 / rho)^(1/5)."""
    if rho is None:
        rho = ism_density(n_per_cc)
    return xi * (E * t * t / rho) ** 0.2


def sedov_velocity(t: float, E: float = E_SN, rho: float = None,
                   n_per_cc: float = 1.0, xi: float = XI_SEDOV) -> float:
    """Sedov shock speed (m/s): v = dR/dt = (2/5) R / t."""
    return 0.4 * sedov_radius(t, E, rho, n_per_cc, xi) / t


def phase(t: float, M_ej: float, v_ej: float, E: float = E_SN,
          n_per_cc: float = 1.0) -> str:
    """Name the SNR evolutionary phase at age t (s): 'free expansion', 'Sedov-Taylor',
    'snowplow', or 'merged', using the sweep-up time, a ~3e4 yr radiative onset, and a
    ~1e6 yr merge time."""
    rho = ism_density(n_per_cc)
    t_free = free_expansion_end_time(M_ej, rho, v_ej)
    t_radiative = 3e4 * YEAR
    t_merge = 1e6 * YEAR
    if t < t_free:
        return "free expansion"
    if t < t_radiative:
        return "Sedov-Taylor"
    if t < t_merge:
        return "snowplow"
    return "merged"


def merge_radius(E: float = E_SN, n_per_cc: float = 1.0,
                 v_merge: float = 10e3) -> float:
    """Radius (m) where the Sedov shock slows to the ISM velocity dispersion v_merge
    (~10 km/s) and the remnant merges: found from v_Sedov(t)=v_merge then R(t)."""
    rho = ism_density(n_per_cc)
    # v = (2/5) xi (E/rho)^(1/5) t^(-3/5) = v_merge  ->  solve for t, then R.
    k = 0.4 * XI_SEDOV * (E / rho) ** 0.2
    t = (k / v_merge) ** (5.0 / 3.0)
    return sedov_radius(t, E, rho)

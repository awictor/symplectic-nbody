"""The Chandrasekhar mass: the maximum mass of a white dwarf.

A white dwarf is held up against gravity by electron degeneracy pressure. When
the electrons become relativistic (in a massive, dense dwarf), the equation of
state stiffens to P proportional to rho^{4/3} -- a polytrope of index n = 3. For
n = 3 the polytrope mass is INDEPENDENT of central density: there is a single
possible mass, and no white dwarf can exceed it. That is the Chandrasekhar mass,

    M_Ch = (2.01824 / (4 pi)) * (sqrt(4 pi) hbar c / G)^{3/2} / (mu_e m_H)^2 ... ,

which reduces to the compact form

    M_Ch ~ 5.83 / mu_e^2  solar masses  ~ 1.44 M_sun  (mu_e = 2 for C/O),

where mu_e is the mean molecular weight per electron and 2.01824 is the n=3
Lane-Emden mass factor -xi_1^2 theta'(xi_1). Above M_Ch the star cannot support
itself and collapses -- the trigger for type-Ia supernovae and the reason
neutron stars and black holes exist.

This module computes M_Ch from fundamental constants (reusing the n=3 Lane-Emden
solution from lane_emden.py) and reproduces the ~1.44 M_sun value. Pure stdlib.
"""

from __future__ import annotations

import math

from lane_emden import solve

# fundamental constants (SI)
HBAR = 1.054571817e-34      # J s
C = 2.99792458e8            # m/s
G = 6.67430e-11             # m^3 kg^-1 s^-2
M_H = 1.6726219e-27         # kg (proton / atomic mass unit ~)
M_SUN = 1.98892e30          # kg

# n=3 Lane-Emden mass factor omega_3 = -xi_1^2 theta'(xi_1) ~ 2.01824
_XIS, _TH, _XI1, OMEGA_3 = solve(3.0)


def chandrasekhar_mass(mu_e: float = 2.0) -> float:
    """Chandrasekhar mass in kg for mean molecular weight per electron mu_e.

    M_Ch = 4 pi omega_3 * ( (hbar c / G)^{3/2} ) / (mu_e m_H)^2  *  (1/(4 pi))^2 ...

    Using the standard relativistic-degenerate polytrope result:
      K = (hbar c / 12 pi^2) (3 pi^2 / (mu_e m_H))^{4/3}   (P = K rho^{4/3})
      M = 4 pi omega_3 [ K / (pi G) ]^{3/2}
    """
    K = (HBAR * C / (12.0 * math.pi ** 2)) * (3.0 * math.pi ** 2 / (mu_e * M_H)) ** (4.0 / 3.0)
    M = 4.0 * math.pi * OMEGA_3 * (K / (math.pi * G)) ** 1.5
    return M


def chandrasekhar_mass_solar(mu_e: float = 2.0) -> float:
    """Chandrasekhar mass in solar masses."""
    return chandrasekhar_mass(mu_e) / M_SUN


def white_dwarf_structure(rho_c: float, mu_e: float = 2.0,
                          dr: float = 1e4, r_max: float = 5e7):
    """Integrate a white dwarf's hydrostatic structure with the FULL relativistic
    degenerate electron equation of state (not a fixed polytrope). Returns
    (radius_m, mass_kg). As rho_c grows the electrons turn relativistic, the star
    shrinks, and the mass asymptotes to the Chandrasekhar value.

    The exact degenerate pressure P(x) and density rho(x) as functions of the
    dimensionless Fermi momentum x = p_F/(m_e c):
        rho = B mu_e x^3,   P = A f(x),
        f(x) = x(2x^2-3) sqrt(x^2+1) + 3 asinh(x),
    with A = pi m_e^4 c^5 / (3 h^3), B = 8 pi m_e^3 c^3 mu_H / (3 h^3)."""
    m_e = 9.1093837e-31
    h = 6.62607015e-34
    A = math.pi * m_e ** 4 * C ** 5 / (3.0 * h ** 3)
    B = 8.0 * math.pi * m_e ** 3 * C ** 3 * M_H / (3.0 * h ** 3)

    def x_of_rho(rho):
        return (rho / (B * mu_e)) ** (1.0 / 3.0) if rho > 0 else 0.0

    def P_of_rho(rho):
        x = x_of_rho(rho)
        f = x * (2 * x * x - 3) * math.sqrt(x * x + 1.0) + 3.0 * math.asinh(x)
        return A * f

    def rho_of_P(P):
        # invert P(rho) by bisection (monotone)
        lo, hi = 1e-6, 1e15
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if P_of_rho(mid) < P:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # integrate dP/dr = -G m rho / r^2, dm/dr = 4 pi r^2 rho
    r = dr
    rho = rho_c
    P = P_of_rho(rho_c)
    m = 4.0 / 3.0 * math.pi * r ** 3 * rho_c
    while r < r_max and P > 1e-8 * P_of_rho(rho_c):
        dP = -G * m * rho / (r * r) * dr
        dm = 4.0 * math.pi * r * r * rho * dr
        P += dP
        m += dm
        r += dr
        if P <= 0:
            break
        rho = rho_of_P(P)
    return r, m

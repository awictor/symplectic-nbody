"""The Parker wind: why the solar corona cannot stay still.

Eugene Parker asked whether a hot corona could sit in hydrostatic equilibrium and
found it could not: an isothermal atmosphere around the Sun has a pressure that stays
finite at infinity, far above the near-zero pressure of interstellar space. The corona
must therefore expand -- a supersonic solar wind is inevitable.

The steady, isothermal, spherically-symmetric wind obeys Parker's equation, and the
physically-correct solution passes smoothly through a sonic critical point where the
flow speed equals the isothermal sound speed c_s = sqrt(k_B T / (mu m_p)). That
critical radius is

    r_c = G M / (2 c_s^2),

inside which the wind is subsonic and outside which it is supersonic. For a ~1-2 MK
corona r_c is a few solar radii, and integrating the wind outward gives a few hundred
km/s at 1 AU -- exactly the solar wind that was confirmed by the Mariner probes,
vindicating Parker over the static-corona camp.

The transonic solution is found from the Parker integral (the Bernoulli-like constant),

    (v/c_s)^2 - ln (v/c_s)^2 = 4 ln (r/r_c) + 4 (r_c/r) - 3,

which is solved here by bisection on the correct branch (subsonic for r<r_c,
supersonic for r>r_c). This module gives the sound speed, the critical radius, the
wind speed at any radius, and the Mach number, and reproduces the few-hundred-km/s
wind at 1 AU. SI units. Pure stdlib; the outflow companion to the Bondi-accretion and
Alfven modules.
"""

from __future__ import annotations

import math

G = 6.67430e-11
K_B = 1.380649e-23
M_P = 1.6726219e-27
M_SUN = 1.989e30
R_SUN = 6.957e8
AU = 1.495978707e11


def sound_speed(T: float, mu: float = 0.6) -> float:
    """Isothermal sound speed c_s = sqrt(k_B T / (mu m_p)) (m/s)."""
    return math.sqrt(K_B * T / (mu * M_P))


def critical_radius(T: float, M: float = M_SUN, mu: float = 0.6) -> float:
    """Sonic critical radius r_c = G M / (2 c_s^2) (m): the wind passes through the
    sound speed here."""
    cs = sound_speed(T, mu)
    return G * M / (2.0 * cs * cs)


def _parker_rhs(r_over_rc: float) -> float:
    """Right-hand side of the Parker integral: 4 ln(r/r_c) + 4 (r_c/r) - 3."""
    x = r_over_rc
    return 4.0 * math.log(x) + 4.0 / x - 3.0


def wind_mach(r: float, T: float, M: float = M_SUN, mu: float = 0.6) -> float:
    """Wind Mach number v/c_s at radius r on the transonic solution, by bisection.
    Subsonic branch inside r_c, supersonic branch outside; =1 at r_c."""
    rc = critical_radius(T, M, mu)
    x = r / rc
    if abs(x - 1.0) < 1e-9:
        return 1.0
    rhs = _parker_rhs(x)
    # solve w - ln w = rhs for w = (v/c_s)^2, picking the correct branch
    if x < 1.0:
        lo, hi = 1e-12, 1.0            # subsonic: w < 1
    else:
        lo, hi = 1.0, 1e6              # supersonic: w > 1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = mid - math.log(mid) - rhs
        # w - ln w is decreasing for w<1, increasing for w>1
        if x < 1.0:
            if f > 0.0:
                lo = mid
            else:
                hi = mid
        else:
            if f < 0.0:
                lo = mid
            else:
                hi = mid
    return math.sqrt(0.5 * (lo + hi))


def wind_speed(r: float, T: float, M: float = M_SUN, mu: float = 0.6) -> float:
    """Wind speed v(r) (m/s) = Mach(r) * c_s on the transonic Parker solution."""
    return wind_mach(r, T, M, mu) * sound_speed(T, mu)

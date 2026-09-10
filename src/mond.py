"""MOND: Modified Newtonian Dynamics -- flat rotation curves without dark matter.

Instead of adding unseen mass (the dark halo of rotation_curve.py), MOND
(Milgrom 1983) modifies gravity itself below a tiny acceleration scale
a0 ~ 1.2e-10 m/s^2. The true acceleration g relates to the Newtonian one g_N by

    g mu(g / a0) = g_N,

with an interpolating function mu(x) -> 1 for x >> 1 (normal Newtonian regime)
and mu(x) -> x for x << 1 (deep-MOND regime). In the deep-MOND limit
g = sqrt(g_N a0), so for a point mass g_N = G M / r^2 gives

    g = sqrt(G M a0) / r   ->   v_circ = (G M a0)^{1/4} = const,

a naturally FLAT rotation curve, and

    v_flat^4 = G M a0        the baryonic Tully-Fisher relation,

which links a galaxy's flat rotation speed to its *visible* mass with almost no
scatter -- MOND's cleanest prediction. This module implements the standard and
simple interpolating functions, solves for g, and reproduces the flat curve and
the Tully-Fisher v^4 ~ M law. SI units. Pure stdlib.
"""

from __future__ import annotations

import math

G = 6.67430e-11
A0 = 1.2e-10                 # MOND acceleration scale, m/s^2
M_SUN = 1.98892e30
KM = 1000.0
KPC = 3.0857e19              # m


def mu_standard(x: float) -> float:
    """Standard interpolating function mu(x) = x / sqrt(1 + x^2)."""
    return x / math.sqrt(1.0 + x * x)


def mu_simple(x: float) -> float:
    """Simple interpolating function mu(x) = x / (1 + x)."""
    return x / (1.0 + x)


def mond_acceleration(g_newton: float, mu=mu_simple) -> float:
    """Solve g mu(g/a0) = g_N for the true acceleration g, given the Newtonian
    g_N. Uses bisection (g is monotone in g_N)."""
    if g_newton <= 0.0:
        return 0.0
    lo, hi = 1e-30, max(g_newton * 10.0, A0 * 10.0)
    for _ in range(200):
        g = 0.5 * (lo + hi)
        if g * mu(g / A0) < g_newton:
            lo = g
        else:
            hi = g
    return 0.5 * (lo + hi)


def circular_speed(M: float, r: float, mu=mu_simple) -> float:
    """MOND circular speed at radius r around baryonic point mass M."""
    g_N = G * M / (r * r)
    g = mond_acceleration(g_N, mu)
    return math.sqrt(g * r)


def deep_mond_vflat(M: float) -> float:
    """Asymptotic flat rotation speed v = (G M a0)^{1/4} (deep-MOND limit)."""
    return (G * M * A0) ** 0.25


def tully_fisher_mass(v_flat: float) -> float:
    """Baryonic mass from the flat speed: M = v_flat^4 / (G a0)."""
    return v_flat ** 4 / (G * A0)

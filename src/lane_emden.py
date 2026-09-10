"""The Lane-Emden equation: the structure of a self-gravitating gas sphere.

A star in hydrostatic equilibrium whose pressure and density follow a polytropic
law P = K rho^{1 + 1/n} has a density profile theta(xi)^n governed by the
dimensionless Lane-Emden equation

    (1/xi^2) d/dxi ( xi^2 dtheta/dxi ) + theta^n = 0,
    theta(0) = 1,  theta'(0) = 0.

Here xi is scaled radius and theta the scaled density; the surface is the first
zero theta(xi_1) = 0. The polytropic index n encodes the equation of state:
n = 1.5 is a fully convective star or a non-relativistic degenerate white dwarf,
n = 3 the Eddington standard model / relativistic degenerate limit.

Three indices have closed-form solutions, which this module reproduces exactly:

    n = 0 : theta = 1 - xi^2/6,          xi_1 = sqrt(6)
    n = 1 : theta = sin(xi)/xi,          xi_1 = pi
    n = 5 : theta = 1/sqrt(1 + xi^2/3),  xi_1 = infinity (infinite radius)

The code integrates the ODE for any n and returns the profile, the surface
radius xi_1, and the derivative there (which sets the star's mass). Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def _derivs(xi: float, theta: float, phi: float, n: float):
    """State (theta, phi=dtheta/dxi). Returns (dtheta, dphi).
    dphi/dxi = -theta^n - 2/xi phi, regularized at xi->0 where dphi -> -1/3."""
    dtheta = phi
    if xi < 1e-8:
        dphi = -1.0 / 3.0  # series limit: theta ~ 1 - xi^2/6
    else:
        base = theta if theta > 0.0 else 0.0   # theta^n undefined for theta<0
        dphi = -(base ** n) - 2.0 / xi * phi
    return dtheta, dphi


def solve(n: float, dxi: float = 1e-4, xi_max: float = 20.0
          ) -> Tuple[List[float], List[float], float, float]:
    """Integrate the Lane-Emden equation for index n with RK4.
    Returns (xis, thetas, xi_1, minus_xi1sq_thetaprime) where xi_1 is the first
    zero (surface) and the last value is -xi_1^2 theta'(xi_1), which sets the
    dimensionless mass. If theta never reaches 0 within xi_max (e.g. n>=5),
    xi_1 is inf."""
    xi, theta, phi = 1e-8, 1.0, 0.0
    xis, thetas = [xi], [theta]
    steps = int(xi_max / dxi)
    for _ in range(steps):
        k1t, k1p = _derivs(xi, theta, phi, n)
        k2t, k2p = _derivs(xi + 0.5 * dxi, theta + 0.5 * dxi * k1t, phi + 0.5 * dxi * k1p, n)
        k3t, k3p = _derivs(xi + 0.5 * dxi, theta + 0.5 * dxi * k2t, phi + 0.5 * dxi * k2p, n)
        k4t, k4p = _derivs(xi + dxi, theta + dxi * k3t, phi + dxi * k3p, n)
        theta_new = theta + dxi / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        phi_new = phi + dxi / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        xi_new = xi + dxi
        if theta_new <= 0.0 < theta:
            # linear interpolation to the surface theta=0
            frac = theta / (theta - theta_new)
            xi_1 = xi + frac * dxi
            phi_1 = phi + frac * (phi_new - phi)
            xis.append(xi_1); thetas.append(0.0)
            return xis, thetas, xi_1, -xi_1 * xi_1 * phi_1
        theta, phi, xi = theta_new, phi_new, xi_new
        xis.append(xi); thetas.append(theta)
    return xis, thetas, float("inf"), float("nan")


# --- exact solutions for validation ---------------------------------------
def analytic_n0(xi: float) -> float:
    return 1.0 - xi * xi / 6.0


def analytic_n1(xi: float) -> float:
    return 1.0 if xi < 1e-12 else math.sin(xi) / xi


def analytic_n5(xi: float) -> float:
    return 1.0 / math.sqrt(1.0 + xi * xi / 3.0)


KNOWN_XI1 = {0.0: math.sqrt(6.0), 1.0: math.pi, 5.0: float("inf")}

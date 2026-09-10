"""Kozai-Lidov cycles: eccentricity <-> inclination oscillations in a triple.

In a hierarchical triple (a tight inner binary orbited by a distant third body),
the inner orbit's eccentricity and inclination undergo large coupled
oscillations when the mutual inclination exceeds a critical value. This is the
Kozai-Lidov mechanism -- it drives high-eccentricity migration of hot Jupiters,
merges black-hole binaries, and flips orbits over.

We work in the standard test-particle quadrupole approximation, integrating the
doubly-averaged (secular) Hamiltonian flow in the canonical pair (omega, G):

    e = sqrt(1 - G^2),   cos i = Theta / G,   Theta = sqrt(1-e^2) cos i  (conserved)

    F(G, omega) = (2 + 3 e^2)(3 cos^2 i - 1) + 15 e^2 sin^2 i cos(2 omega)

with omega the inner argument of periapsis. Hamilton's equations
domega/dtau = dF/dG, dG/dtau = -dF/domega are integrated with RK4; the
G-derivative is taken numerically so there are no hand-differentiation errors.

Two exact predictions this reproduces:
  * conserved Theta = sqrt(1-e^2) cos i (the z-angular-momentum of the inner orbit),
  * starting near-circular at inclination i0, the maximum eccentricity reached is
    e_max = sqrt(1 - (5/3) cos^2 i0), real only for i0 > arccos(sqrt(3/5)) ~ 39.2 deg
    -- the Kozai critical angle. Below it, no eccentricity is excited.

Pure stdlib.
"""

from __future__ import annotations

import math
from typing import List, Tuple

CRITICAL_ANGLE_DEG = math.degrees(math.acos(math.sqrt(3.0 / 5.0)))  # ~39.23


def _F(G: float, omega: float, Theta: float) -> float:
    e2 = max(0.0, 1.0 - G * G)
    cos2i = min(1.0, (Theta / G) ** 2)
    sin2i = 1.0 - cos2i
    return (2.0 + 3.0 * e2) * (3.0 * cos2i - 1.0) + 15.0 * e2 * sin2i * math.cos(2.0 * omega)


def _dF_dG(G: float, omega: float, Theta: float, h: float = 1e-7) -> float:
    return (_F(G + h, omega, Theta) - _F(G - h, omega, Theta)) / (2.0 * h)


def _dF_domega(G: float, omega: float, Theta: float) -> float:
    # analytic in omega: d/domega [15 e^2 sin^2 i cos 2omega] = -30 e^2 sin^2 i sin 2omega
    e2 = max(0.0, 1.0 - G * G)
    sin2i = 1.0 - min(1.0, (Theta / G) ** 2)
    return -30.0 * e2 * sin2i * math.sin(2.0 * omega)


def evolve(e0: float, i0_deg: float, omega0_deg: float = 90.0,
           dtau: float = 1e-3, n_steps: int = 200000, sample_every: int = 200):
    """Integrate the secular Kozai flow. Returns (tau, e, i_deg, F) samples."""
    i0 = math.radians(i0_deg)
    G = math.sqrt(1.0 - e0 * e0)
    Theta = G * math.cos(i0)              # conserved
    omega = math.radians(omega0_deg)

    # Hamilton's equations for the canonical pair (omega, G):
    #   d omega/dtau = dF/dG ,   dG/dtau = -dF/domega
    def rhs(G, w):
        return _dF_dG(G, w, Theta), -_dF_domega(G, w, Theta)  # (d omega, d G)

    taus, es, idegs, Fs = [0.0], [e0], [i0_deg], [_F(G, omega, Theta)]
    tau = 0.0
    for s in range(n_steps):
        a1w, a1G = rhs(G, omega)
        a2w, a2G = rhs(G + 0.5 * dtau * a1G, omega + 0.5 * dtau * a1w)
        a3w, a3G = rhs(G + 0.5 * dtau * a2G, omega + 0.5 * dtau * a2w)
        a4w, a4G = rhs(G + dtau * a3G, omega + dtau * a3w)
        omega += dtau / 6.0 * (a1w + 2 * a2w + 2 * a3w + a4w)
        G += dtau / 6.0 * (a1G + 2 * a2G + 2 * a3G + a4G)
        G = min(1.0, max(abs(Theta) + 1e-12, G))  # keep in physical range
        tau += dtau
        if s % sample_every == 0:
            e = math.sqrt(max(0.0, 1.0 - G * G))
            cosi = max(-1.0, min(1.0, Theta / G))
            taus.append(tau); es.append(e); idegs.append(math.degrees(math.acos(cosi)))
            Fs.append(_F(G, omega, Theta))
    return taus, es, idegs, Fs


def theta_of(e: float, i_deg: float) -> float:
    """The conserved Kozai integral Theta = sqrt(1-e^2) cos i."""
    return math.sqrt(1.0 - e * e) * math.cos(math.radians(i_deg))


def analytic_emax(i0_deg: float) -> float:
    """Maximum eccentricity from a near-circular start at inclination i0:
    e_max = sqrt(1 - (5/3) cos^2 i0). Returns 0 below the critical angle."""
    val = 1.0 - (5.0 / 3.0) * math.cos(math.radians(i0_deg)) ** 2
    return math.sqrt(val) if val > 0.0 else 0.0

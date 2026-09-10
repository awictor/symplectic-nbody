"""Orbits around a Schwarzschild black hole (strong-field general relativity).

Outside a non-rotating black hole of mass M, a test particle's radial motion
obeys (geometrized units G = c = 1, so a length is measured in units of M):

    (dr/dtau)^2 = E^2 - V_eff(r),   V_eff(r) = (1 - 2M/r)(1 + L^2/r^2)

for a massive particle (the "+1"), or without the +1 for a photon. Two famous
features drop out of V_eff:

  * the INNERMOST STABLE CIRCULAR ORBIT (ISCO) at r = 6M -- inside it no stable
    circular orbit exists and matter plunges in (sets the inner edge of accretion
    disks and the black-hole spin measurement);
  * the PHOTON SPHERE at r = 3M -- the radius where light orbits in a circle
    (the ring seen in the M87*/Sgr A* images).

Bound orbits precess (perihelion advance far larger than Newtonian), and orbits
that pass inside the potential barrier plunge to r = 0. This module builds
V_eff, locates the circular orbits, integrates the orbit shape r(phi) via the
u = 1/r Binet-style equation, and measures the per-orbit precession.

Pure stdlib; distances in units of M.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def V_eff(r: float, L: float, M: float = 1.0, massive: bool = True) -> float:
    """Schwarzschild effective potential per unit mass (massive) or the photon
    potential (massive=False)."""
    rest = 1.0 if massive else 0.0
    return (1.0 - 2.0 * M / r) * (rest + L * L / (r * r))


def isco_radius(M: float = 1.0) -> float:
    """Innermost stable circular orbit radius = 6M for Schwarzschild."""
    return 6.0 * M


def photon_sphere_radius(M: float = 1.0) -> float:
    """Photon sphere radius = 3M."""
    return 3.0 * M


def circular_orbit_L(r: float, M: float = 1.0) -> float:
    """Specific angular momentum of a circular orbit at radius r (massive):
    L^2 = M r^2 / (r - 3M). Real only for r > 3M (inside 3M no circular orbit)."""
    denom = r - 3.0 * M
    if denom <= 0.0:
        return float("nan")
    return math.sqrt(M * r * r / denom)


def orbit_shape(r0: float, L: float, E: float, M: float = 1.0,
                dphi: float = 1e-4, max_phi: float = 200.0,
                r_plunge: float = 1e-2):
    """Integrate the orbit shape u(phi) = 1/r(phi) with the Schwarzschild Binet
    equation  d^2u/dphi^2 + u = M/L^2 + 3 M u^2.
    Starts at r0 with du/dphi = 0 (an apsis). Returns (phis, rs); stops at plunge
    (r -> 0) or when max_phi is reached."""
    u = 1.0 / r0
    dudphi = 0.0
    phi = 0.0
    phis, rs = [0.0], [r0]
    n = int(max_phi / dphi)
    for _ in range(n):
        # RK4 on (u, du/dphi)
        def ddu(u):
            return M / (L * L) + 3.0 * M * u * u - u

        k1u, k1d = dudphi, ddu(u)
        k2u, k2d = dudphi + 0.5 * dphi * k1d, ddu(u + 0.5 * dphi * k1u)
        k3u, k3d = dudphi + 0.5 * dphi * k2d, ddu(u + 0.5 * dphi * k2u)
        k4u, k4d = dudphi + dphi * k3d, ddu(u + dphi * k3u)
        u += dphi / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        dudphi += dphi / 6.0 * (k1d + 2 * k2d + 2 * k3d + k4d)
        phi += dphi
        if u <= 0.0:           # r -> infinity (unbound escape)
            break
        r = 1.0 / u
        if r < r_plunge:       # plunged into the hole
            phis.append(phi); rs.append(r)
            break
        if len(phis) == 0 or (phi - phis[-1]) >= 0.01:
            phis.append(phi); rs.append(r)
    return phis, rs


def precession_per_orbit(r_peri: float, r_apo: float, M: float = 1.0) -> float:
    """Relativistic apsidal advance per radial oscillation (radians), from the
    orbit's angular period. Integrates u(phi) from one perihelion to the next and
    returns (delta phi - 2*pi)."""
    # angular momentum & energy from the two turning points (V_eff equal there)
    # solve E^2 = V(r_peri) = V(r_apo) for L^2:
    #   (1-2M/rp)(1+L^2/rp^2) = (1-2M/ra)(1+L^2/ra^2)
    fp, fa = 1.0 - 2.0 * M / r_peri, 1.0 - 2.0 * M / r_apo
    # fp + fp L^2/rp^2 = fa + fa L^2/ra^2
    num = fp - fa
    den = fa / (r_apo ** 2) - fp / (r_peri ** 2)
    L2 = num / den
    L = math.sqrt(L2)
    # integrate from perihelion outward until the next perihelion (u returns to
    # its start value having gone through apohelion)
    u = 1.0 / r_peri
    dudphi = 0.0
    dphi = 1e-4
    phi = 0.0

    def ddu(u):
        return M / L2 + 3.0 * M * u * u - u

    # step once to leave the apsis, then run until du/dphi returns to 0 the 2nd time
    zero_crossings = 0
    prev_d = dudphi
    for _ in range(int(100.0 / dphi)):
        k1u, k1d = dudphi, ddu(u)
        k2u, k2d = dudphi + 0.5 * dphi * k1d, ddu(u + 0.5 * dphi * k1u)
        k3u, k3d = dudphi + 0.5 * dphi * k2d, ddu(u + 0.5 * dphi * k2u)
        k4u, k4d = dudphi + dphi * k3d, ddu(u + dphi * k3u)
        u += dphi / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        dudphi += dphi / 6.0 * (k1d + 2 * k2d + 2 * k3d + k4d)
        phi += dphi
        if prev_d < 0.0 <= dudphi or prev_d > 0.0 >= dudphi:
            zero_crossings += 1
            if zero_crossings == 2:   # perihelion -> apohelion -> perihelion
                break
        prev_d = dudphi
    return phi - 2.0 * math.pi

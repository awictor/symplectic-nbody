"""Galaxy rotation curves and the evidence for dark matter.

A star on a circular orbit at radius r feels only the mass enclosed within r
(Newton's shell theorem), so its circular speed is

    v_c(r) = sqrt( G M(<r) / r ).

If a galaxy were just its visible disk, M(<r) would level off past the luminous
edge and v_c would fall as r^{-1/2} -- a "Keplerian" decline. Instead, observed
rotation curves stay FLAT far beyond the visible disk. That flatness is the
classic evidence for an extended dark-matter halo whose enclosed mass keeps
growing as M(<r) ~ r.

This module builds the circular-speed curve from three mass components:
  * an exponential stellar disk,
  * (optionally) a compact bulge,
  * an NFW dark-matter halo,
and shows the visible-only curve declining while disk+halo stays flat.

Units are left generic (set G); pure stdlib.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple


def disk_enclosed_mass(r: float, M_disk: float, R_d: float) -> float:
    """Enclosed mass of a razor-thin exponential disk approximated as a spherical
    mass distribution with the same cumulative profile:
        M(<r) = M_disk * [1 - (1 + r/R_d) exp(-r/R_d)].
    (Exact for a spherical exponential; a good stand-in for the disk's M(<r).)"""
    x = r / R_d
    return M_disk * (1.0 - (1.0 + x) * math.exp(-x))


def nfw_enclosed_mass(r: float, rho_s: float, r_s: float) -> float:
    """NFW dark-halo enclosed mass:
        M(<r) = 4 pi rho_s r_s^3 [ ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s) ].
    Grows ~ r at large radius, which is what keeps the rotation curve flat."""
    x = r / r_s
    return 4.0 * math.pi * rho_s * r_s ** 3 * (math.log(1.0 + x) - x / (1.0 + x))


def circular_speed(r: float, mass_enclosed: float, G: float = 1.0) -> float:
    """v_c = sqrt(G M(<r) / r)."""
    if r <= 0.0:
        return 0.0
    return math.sqrt(G * mass_enclosed / r)


def rotation_curve(radii: List[float], M_disk: float, R_d: float,
                   halo: Tuple[float, float] = None, G: float = 1.0):
    """Return (v_visible, v_total) speed lists over `radii`. v_visible uses the
    disk only; v_total adds the NFW halo (rho_s, r_s) if given."""
    v_vis, v_tot = [], []
    for r in radii:
        m_disk = disk_enclosed_mass(r, M_disk, R_d)
        v_vis.append(circular_speed(r, m_disk, G))
        m_tot = m_disk
        if halo is not None:
            m_tot += nfw_enclosed_mass(r, halo[0], halo[1])
        v_tot.append(circular_speed(r, m_tot, G))
    return v_vis, v_tot


def keplerian_tail_slope(radii: List[float], speeds: List[float]) -> float:
    """Log-log slope d ln v / d ln r over the OUTER half of the curve. A pure
    Keplerian decline gives -1/2; a flat curve gives ~0."""
    n = len(radii)
    lo = n // 2
    lr = [math.log(radii[i]) for i in range(lo, n)]
    lv = [math.log(speeds[i]) for i in range(lo, n)]
    m = len(lr)
    mr, mv = sum(lr) / m, sum(lv) / m
    num = sum((lr[i] - mr) * (lv[i] - mv) for i in range(m))
    den = sum((lr[i] - mr) ** 2 for i in range(m))
    return num / den

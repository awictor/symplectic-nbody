"""Photon-photon pair production: why the gamma-ray sky has a horizon.

Two photons can collide and turn into an electron-positron pair, gamma + gamma
-> e+ + e-, if they carry enough center-of-mass energy. For photons of energies
E1, E2 meeting at angle theta the threshold is

    E1 E2 (1 - cos theta) >= 2 (m_e c^2)^2,

so a head-on collision (theta = 180 deg) needs E1 E2 >= (m_e c^2)^2 = (511 keV)^2.
Two 511 keV gamma rays can just make a pair head-on; a single high-energy gamma
ray pair-produces off a low-energy background photon when

    E_gamma >= (m_e c^2)^2 / E_background.

That is why TeV gamma rays from distant blazars are absorbed by the infrared/
optical extragalactic background light, and PeV photons by the CMB -- the
universe is opaque to gamma rays beyond a horizon that shrinks with energy.

This module gives the threshold condition, the required partner energy, and the
head-on threshold, and reproduces the 511 keV and TeV-on-IR cases. SI units,
energies in joules with eV/keV/TeV helpers. Pure stdlib; the m_e c^2 scale ties
it to Compton scattering.
"""

from __future__ import annotations

import math

M_E = 9.1093837e-31
C = 2.99792458e8
EV = 1.602176634e-19
KEV = 1e3 * EV
MEV = 1e6 * EV
GEV = 1e9 * EV
TEV = 1e12 * EV

MEC2 = M_E * C * C            # electron rest energy, ~511 keV in joules


def electron_rest_energy_kev() -> float:
    """m_e c^2 in keV (~511)."""
    return MEC2 / KEV


def can_pair_produce(E1: float, E2: float, theta_rad: float = math.pi) -> bool:
    """True if two photons of energies E1, E2 (joules) meeting at angle theta can
    make an e+e- pair: E1 E2 (1 - cos theta) >= 2 (m_e c^2)^2."""
    return E1 * E2 * (1.0 - math.cos(theta_rad)) >= 2.0 * MEC2 * MEC2


def head_on_threshold() -> float:
    """For equal-energy head-on photons, the threshold energy each (joules):
    E = m_e c^2 (so two 511 keV gammas just make a pair)."""
    return MEC2


def partner_threshold_energy(E_background: float) -> float:
    """Minimum gamma-ray energy (joules) to pair-produce head-on off a background
    photon of energy E_background: E_gamma = (m_e c^2)^2 / E_background."""
    return MEC2 * MEC2 / E_background


def gamma_ray_horizon_case(E_gamma_tev: float, background_ev: float) -> bool:
    """Does a gamma ray of E_gamma (TeV) get absorbed head-on by a background
    photon of the given energy (eV)? True -> the universe is opaque to it."""
    return can_pair_produce(E_gamma_tev * TEV, background_ev * EV)

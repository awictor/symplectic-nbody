"""The Stromgren sphere: the glowing bubble of ionized gas around a hot star.

A hot, massive star floods its surroundings with ultraviolet photons energetic enough
to ionize hydrogen (above 13.6 eV). Those photons carve out a sphere of ionized gas --
an HII region -- embedded in the neutral cloud. The sphere's size is set by a simple
balance: in equilibrium, every ionizing photon the star emits is eventually used up
re-ionizing an atom that has recombined. So the total recombination rate inside the
sphere equals the star's ionizing-photon output Q:

    Q = (4/3) pi R_s^3 * n^2 * alpha_B,

giving the Stromgren radius

    R_s = ( 3 Q / (4 pi n^2 alpha_B) )^(1/3),

where n is the hydrogen number density and alpha_B ~ 2.6e-13 cm^3/s is the case-B
recombination coefficient. The R ~ Q^(1/3) and R ~ n^(-2/3) scalings mean a single O
star ionizes a bubble ~1-10 pc across in a typical cloud, while denser gas confines it
to a compact HII region. Inside, the gas is almost fully ionized; the transition to
neutral at the edge is sharp (an ionization front only a mean-free-path thick).

These are the pink emission-line nebulae -- Orion, the Rosette, the Eagle -- that trace
recent massive-star formation, and the same physics set the size of ionized bubbles
during cosmic reionization.

This module gives the Stromgren radius, its scalings, the enclosed recombination rate,
and the ionized-mass, and reproduces the ~pc-scale HII region around an O star. SI
units. Pure stdlib; the ionization companion to the Saha and recombination modules.
"""

from __future__ import annotations

import math

PC = 3.0856775814913673e16     # parsec (m)
M_SUN = 1.989e30
M_P = 1.6726219e-27
ALPHA_B = 2.6e-19              # case-B recombination coefficient (m^3/s), ~2.6e-13 cm^3/s

# representative ionizing-photon outputs Q (photons/s)
Q_O5 = 5e49                   # early O5 star
Q_B0 = 1e48                   # B0 star


def stromgren_radius(Q: float, n: float, alpha_B: float = ALPHA_B) -> float:
    """Stromgren radius (m): R_s = (3 Q / (4 pi n^2 alpha_B))^(1/3). n is the hydrogen
    number density (m^-3), Q the ionizing-photon rate (s^-1)."""
    return (3.0 * Q / (4.0 * math.pi * n * n * alpha_B)) ** (1.0 / 3.0)


def stromgren_radius_pc(Q: float, n_per_cc: float,
                        alpha_B: float = ALPHA_B) -> float:
    """Stromgren radius in parsecs, with density given in atoms per cubic centimetre."""
    n = n_per_cc * 1e6
    return stromgren_radius(Q, n, alpha_B) / PC


def recombination_rate(R: float, n: float, alpha_B: float = ALPHA_B) -> float:
    """Total case-B recombination rate inside a sphere of radius R (s^-1):
    (4/3) pi R^3 n^2 alpha_B. In equilibrium this equals the ionizing output Q."""
    return 4.0 / 3.0 * math.pi * R ** 3 * n * n * alpha_B


def ionized_mass(R: float, n: float, m: float = M_P) -> float:
    """Mass of ionized hydrogen inside the Stromgren sphere (kg):
    (4/3) pi R^3 n m."""
    return 4.0 / 3.0 * math.pi * R ** 3 * n * m


def recombination_time(n: float, alpha_B: float = ALPHA_B) -> float:
    """Recombination timescale t_rec = 1 / (n alpha_B) (s): how fast the sphere
    re-neutralizes if the star switches off."""
    return 1.0 / (n * alpha_B)

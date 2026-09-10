"""The London equations: how a superconductor expels magnetic field.

A superconductor is not just a perfect conductor -- it actively pushes magnetic field out of
its interior, the Meissner effect (1933). A perfect conductor would merely freeze whatever
field was already inside; a superconductor expels it entirely, which is why a magnet levitates
above one. The Londons explained it in 1935: the supercurrent responds to the field so as to
screen it, and the field decays exponentially into the surface,

    B(x) = B0 exp(-x / lambda_L),

over the London penetration depth

    lambda_L = sqrt(m / (mu0 n_s q^2)),

set by the density n_s of superconducting carriers (Cooper pairs, charge 2e). It is tens to
hundreds of nanometres -- so thin films thinner than lambda_L never fully expel the field.

Two length scales then classify a superconductor: lambda_L and the coherence length xi (the
size of a Cooper pair). Their ratio is the Ginzburg-Landau parameter kappa = lambda_L / xi:

    kappa < 1/sqrt(2):  type I -- expels field until it abruptly goes normal at H_c
    kappa > 1/sqrt(2):  type II -- lets field in as quantized flux vortices between H_c1 and H_c2,

which is why the type-II superconductors (Nb-Ti, high-T_c) used in MRI and fusion magnets
survive enormous fields. Above the thermodynamic critical field H_c the condensation energy is
overwhelmed and superconductivity is destroyed.

This module gives the London penetration depth, the exponential field profile, the screening
current, the Ginzburg-Landau parameter and type classification, and the flux per vortex, and
reproduces the ~40 nm penetration depth of a typical metal and the type-I/II boundary. SI
units. Pure stdlib; the superconductivity companion to the BCS and Josephson notes.
"""

from __future__ import annotations

import math

MU0 = 1.25663706212e-6        # vacuum permeability (H/m)
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
H = 6.62607015e-34
KAPPA_C = 1.0 / math.sqrt(2.0)   # type-I / type-II boundary
FLUX_QUANTUM_SC = H / (2.0 * E_CHARGE)


def penetration_depth(carrier_density: float, mass: float = 2 * M_E,
                      charge: float = 2 * E_CHARGE) -> float:
    """London penetration depth lambda_L = sqrt(m / (mu0 n_s q^2)) (m). Default carrier is a
    Cooper pair (mass 2 m_e, charge 2e). Tens to hundreds of nm for real superconductors."""
    return math.sqrt(mass / (MU0 * carrier_density * charge * charge))


def field_profile(x: float, b0: float, penetration: float) -> float:
    """Magnetic field B(x) = B0 exp(-x / lambda_L) (T) a depth x into the superconductor from
    the surface. Screened to 1/e of the surface value at x = lambda_L."""
    return b0 * math.exp(-x / penetration)


def screening_depth_fraction(x: float, penetration: float) -> float:
    """Fraction of the surface field remaining at depth x: exp(-x/lambda_L). ~0.37 at one
    penetration depth, ~0.007 at five."""
    return math.exp(-x / penetration)


def ginzburg_landau_parameter(penetration: float, coherence_length: float) -> float:
    """Ginzburg-Landau parameter kappa = lambda_L / xi: the ratio of penetration depth to
    coherence length that sets the superconductor type."""
    return penetration / coherence_length


def is_type_ii(kappa: float) -> bool:
    """True if the superconductor is type II (kappa > 1/sqrt(2) ~ 0.707): it admits quantized
    flux vortices and survives high fields. Type I (kappa < 1/sqrt(2)) expels field until it
    goes normal."""
    return kappa > KAPPA_C


def vortex_flux() -> float:
    """Magnetic flux carried by a single vortex in a type-II superconductor: the
    superconducting flux quantum h/2e ~ 2.07e-15 Wb. Field enters only in these units."""
    return FLUX_QUANTUM_SC


def critical_field_ratio(kappa: float) -> float:
    """Ratio of upper to lower critical field H_c2/H_c1 for a type-II superconductor, roughly
    ~ 2 kappa^2 / ln(kappa) for large kappa (order-of-magnitude); grows with kappa so
    strongly type-II materials tolerate huge fields."""
    if kappa <= 1.0:
        return 1.0
    return 2.0 * kappa * kappa / math.log(kappa)

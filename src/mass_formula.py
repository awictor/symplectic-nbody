"""The semi-empirical mass formula: nuclear binding from a liquid drop.

Weizsacker modelled the nucleus as a charged liquid drop and wrote its binding energy
as a sum of competing terms:

    B(Z, A) = a_V A                     (volume: each nucleon bonds to neighbours)
            - a_S A^(2/3)               (surface: fewer bonds at the edge)
            - a_C Z(Z-1) / A^(1/3)      (Coulomb: proton repulsion)
            - a_A (A - 2Z)^2 / A        (asymmetry: neutrons != protons costs energy)
            + delta                     (pairing: even-even nuclei are more bound)

The volume term wants a big nucleus; the surface and Coulomb terms penalize it; the
asymmetry term wants N = Z but Coulomb pushes toward neutron excess in heavy nuclei.
Their balance gives the famous binding-energy-per-nucleon curve, rising steeply for
light nuclei, peaking near iron-56 at ~8.8 MeV/nucleon, then declining -- which is
exactly why FUSION releases energy up to iron and FISSION releases it beyond.

Minimizing the mass over Z at fixed A gives the most stable charge (the valley of
stability), and the same formula predicts the energy release of fission and fusion.

This module gives each term, the total binding energy, the binding energy per nucleon,
and the most-stable Z for a given mass number, and reproduces the ~8.8 MeV/nucleon iron
peak. Energies in MeV. Pure stdlib; the nuclear-structure companion to the
radioactive-decay and Gamow modules.
"""

from __future__ import annotations

import math

# Weizsacker coefficients (MeV), a standard set
A_VOLUME = 15.75
A_SURFACE = 17.8
A_COULOMB = 0.711
A_ASYMMETRY = 23.7
A_PAIRING = 11.18


def pairing_term(Z: int, A: int) -> float:
    """Pairing energy (MeV): +delta for even-even, -delta for odd-odd, 0 otherwise,
    with delta = a_P / sqrt(A)."""
    N = A - Z
    delta = A_PAIRING / math.sqrt(A)
    if Z % 2 == 0 and N % 2 == 0:
        return +delta
    if Z % 2 == 1 and N % 2 == 1:
        return -delta
    return 0.0


def binding_energy(Z: int, A: int) -> float:
    """Total nuclear binding energy B(Z, A) in MeV from the semi-empirical mass
    formula (volume - surface - Coulomb - asymmetry + pairing)."""
    volume = A_VOLUME * A
    surface = A_SURFACE * A ** (2.0 / 3.0)
    coulomb = A_COULOMB * Z * (Z - 1) / A ** (1.0 / 3.0)
    asymmetry = A_ASYMMETRY * (A - 2 * Z) ** 2 / A
    return volume - surface - coulomb - asymmetry + pairing_term(Z, A)


def binding_per_nucleon(Z: int, A: int) -> float:
    """Binding energy per nucleon B/A (MeV/nucleon): peaks near iron-56."""
    return binding_energy(Z, A) / A


def most_stable_Z(A: int) -> int:
    """The proton number Z that maximizes binding at fixed mass number A (the valley
    of stability). Found by scanning Z = 1..A."""
    best_Z, best_B = 1, -1e30
    for Z in range(1, A + 1):
        B = binding_energy(Z, A)
        if B > best_B:
            best_B, best_Z = B, Z
    return best_Z


def most_stable_Z_continuous(A: int) -> float:
    """Closed-form continuous estimate of the stable Z (before rounding):
    Z ~ A / (2 + a_C A^(2/3) / (2 a_A)). Drifts below A/2 for heavy nuclei."""
    return A / (2.0 + A_COULOMB * A ** (2.0 / 3.0) / (2.0 * A_ASYMMETRY))

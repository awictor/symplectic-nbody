"""The Stark effect: splitting spectral lines with an electric field.

The electric analogue of the Zeeman effect: put an atom in an electric field and its energy
levels shift, splitting spectral lines. But the response is not the same for every atom.
Hydrogen (and hydrogen-like, degenerate levels) shows a *linear* Stark effect -- a shift
proportional to the field -- because the degenerate states mix into ones with a permanent
electric dipole:

    delta_E = (3/2) n q E a0 k / Z,

for principal quantum number n, field E, and a parabolic quantum number k running from
-(n-1) to +(n-1) in steps. Most atoms have no permanent dipole and instead show the *quadratic*
Stark effect -- a shift proportional to E^2 and to the atomic polarizability alpha:

    delta_E = -(1/2) alpha E^2,

which always lowers the ground-state energy (the field induces a dipole that aligns with it).
Push the field high enough and it can strip the electron entirely: the classical field-
ionization threshold for a level of binding energy E_bind is

    E_ion = E_bind^2 / (4 q^3 / (4 pi eps0)^2)   ~   E_bind^2 (SI-scaled),

which is why Rydberg atoms (huge n, tiny binding) ionize in modest laboratory fields. The
Stark effect broadens spectral lines in dense plasmas (Stark broadening is a standard density
diagnostic) and underlies Stark-tuned spectroscopy.

This module gives the linear and quadratic Stark shifts, the hydrogen linear splitting
pattern, the induced dipole moment, and the field-ionization threshold, and reproduces
hydrogen's field-linear split and the quadratic ground-state shift. SI units. Pure stdlib;
the atomic-field companion to the Zeeman and Bohr notes.
"""

from __future__ import annotations

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
A0 = 5.29177210903e-11        # Bohr radius (m)
RYDBERG_J = 2.1798723611e-18  # Rydberg energy (J)
E_ATOMIC_FIELD = 5.14220674e11  # atomic unit of electric field (V/m)


def linear_stark_shift(n: int, k: int, field: float, z: int = 1) -> float:
    """Linear Stark energy shift delta_E = (3/2) n k q E a0 / Z (J) for a hydrogen-like level
    (n, parabolic k). Proportional to the field -- the hallmark of degenerate levels."""
    return 1.5 * n * k * E_CHARGE * field * A0 / z


def linear_stark_pattern(n: int, field: float, z: int = 1) -> list:
    """The set of linear Stark shifts (J) for level n: k runs from -(n-1) to +(n-1) in unit
    steps, giving 2n-1 equally spaced components symmetric about zero."""
    ks = list(range(-(n - 1), n))
    return [linear_stark_shift(n, k, field, z) for k in ks]


def quadratic_stark_shift(polarizability: float, field: float) -> float:
    """Quadratic Stark shift delta_E = -(1/2) alpha E^2 (J) for a non-degenerate level of
    polarizability alpha. Always negative (the induced dipole lowers the energy)."""
    return -0.5 * polarizability * field * field


def induced_dipole(polarizability: float, field: float) -> float:
    """Induced electric dipole moment p = alpha E (C m) the field creates in the atom."""
    return polarizability * field


def field_ionization_threshold(binding_energy: float) -> float:
    """Classical field-ionization threshold field (V/m). In atomic units F_ion = 1/(16 n^4)
    and E_bind = 1/(2 n^2) Hartree, so F_ion = (E_bind / (2 Ry))^2 / 4 in atomic field units.
    Scaled to SI: F = E_atomic (E_bind / (2 Ry))^2 / 4. Ground-state H ~ 3e11 V/m; Rydberg
    atoms ionize at tiny fields."""
    e_hartree = binding_energy / (2.0 * RYDBERG_J)       # binding in Hartree
    f_atomic = e_hartree * e_hartree / 4.0               # threshold in atomic field units
    return f_atomic * E_ATOMIC_FIELD


def hydrogen_binding_energy(n: int, z: int = 1) -> float:
    """Binding energy |E_n| = Z^2 Rydberg / n^2 (J) of a hydrogen-like level, for the
    ionization-threshold estimate."""
    return z * z * RYDBERG_J / (n * n)

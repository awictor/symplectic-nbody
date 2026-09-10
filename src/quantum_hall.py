"""The quantum Hall effect: resistance quantized to fundamental constants.

Cool a two-dimensional electron gas to near absolute zero, put it in a strong perpendicular
magnetic field, and its Hall resistance stops rising smoothly with field. Instead it locks
onto a staircase of flat plateaus at values that depend on nothing about the material -- not
its size, purity, or carrier density -- but only on fundamental constants:

    R_xy = h / (nu e^2) = R_K / nu,        R_K = h / e^2 = 25812.807 ohm,

where nu is an integer (the integer quantum Hall effect) and R_K the von Klitzing constant.
On each plateau the longitudinal resistance drops to essentially zero -- the current flows
without dissipation along the sample edges. Klaus von Klitzing found it in 1980 and won the
Nobel Prize; because R_K is exact and reproducible to parts per billion across any device, it
is now the definition of the ohm.

The physics: in 2D the electron energies collapse into discrete Landau levels spaced by the
cyclotron energy hbar omega_c = hbar e B / m, each holding e B / h states per unit area. When
exactly nu Landau levels are filled the bulk is gapped (insulating) and only chiral edge
channels conduct, one per filled level, each contributing a conductance e^2/h -- hence the
quantized R_xy. The fractional quantum Hall effect extends this to fractional nu through
electron-electron interactions.

This module gives the von Klitzing constant, the quantized Hall resistance and conductance,
the cyclotron frequency and energy, the Landau-level degeneracy, and the filling factor from
field and density, and reproduces the 25.8 kOhm quantum and the nu=1,2,3 plateaus. SI units.
Pure stdlib; the topological-transport companion to the Hall-effect and Josephson notes.
"""

from __future__ import annotations

E_CHARGE = 1.602176634e-19
H = 6.62607015e-34
HBAR = 1.054571817e-34
M_E = 9.1093837015e-31

VON_KLITZING = H / E_CHARGE ** 2           # R_K = h/e^2 ~ 25812.807 ohm


def von_klitzing_constant() -> float:
    """The von Klitzing constant R_K = h / e^2 ~ 25812.807 ohm: the resistance quantum that
    now defines the SI ohm."""
    return VON_KLITZING


def hall_resistance(filling_factor: int) -> float:
    """Quantized Hall resistance R_xy = R_K / nu (ohm) on the nu-th plateau. 25.8 kOhm at
    nu=1, 12.9 kOhm at nu=2, ..."""
    return VON_KLITZING / filling_factor


def hall_conductance(filling_factor: int) -> float:
    """Quantized Hall conductance sigma_xy = nu e^2 / h (S): nu conductance quanta, one per
    filled Landau level (chiral edge channel)."""
    return filling_factor * E_CHARGE ** 2 / H


def cyclotron_frequency(field: float, mass: float = M_E) -> float:
    """Cyclotron angular frequency omega_c = e B / m (rad/s): the orbit rate that sets the
    Landau-level spacing."""
    return E_CHARGE * field / mass


def landau_level_spacing(field: float, mass: float = M_E) -> float:
    """Landau-level energy spacing hbar omega_c = hbar e B / m (J). Large B and low
    temperature (k_B T << this) are needed to resolve the plateaus."""
    return HBAR * cyclotron_frequency(field, mass)


def landau_degeneracy(field: float) -> float:
    """States per unit area in each Landau level, e B / h (1/m^2): the number of electrons one
    filled level can hold. Equals the flux density in units of the flux quantum."""
    return E_CHARGE * field / H


def filling_factor(density: float, field: float) -> float:
    """Filling factor nu = n h / (e B): the number of Landau levels filled by an areal
    electron density n at field B. Integer nu gives a quantized Hall plateau."""
    return density * H / (E_CHARGE * field)

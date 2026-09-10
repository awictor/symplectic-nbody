"""Nuclear Q-value: the energy released when nuclei rearrange.

Einstein's E = mc^2 says a nuclear reaction releases (or absorbs) energy equal to the
change in rest mass:

    Q = (m_reactants - m_products) c^2.

A positive Q means the products are lighter than the reactants -- the missing mass came
out as kinetic energy and radiation (exothermic). The mass "defect" is tiny -- a fraction
of a percent -- but c^2 is enormous, so the energies are millions of times larger than
chemical bonds.

Fusion of light nuclei toward iron is exothermic: deuterium + tritium -> helium-4 +
neutron releases 17.6 MeV, the reaction powering hydrogen bombs and tokamaks. Fission of
heavy nuclei is also exothermic: U-235 + n splits with ~200 MeV per event, the energy
of reactors and atomic bombs. Both climb the binding-energy curve toward the iron peak.
A convenient shortcut is that 1 atomic mass unit of defect equals 931.494 MeV.

This module gives the Q-value from a mass change (in kg, in atomic mass units, or from
binding energies), the fraction of rest mass converted, and the energy yield per
kilogram of fuel, and reproduces the 17.6 MeV D-T fusion and ~200 MeV U-235 fission.
SI units, energies via a MeV helper. Pure stdlib; the mass-energy companion to the
mass-formula and Gamow modules.
"""

from __future__ import annotations

import math

C = 2.99792458e8
AMU = 1.66053906660e-27
MEV = 1e6 * 1.602176634e-19
AMU_TO_MEV = 931.49410242      # energy equivalent of one atomic mass unit (MeV)
N_A = 6.02214076e23


def q_value_from_mass_change(delta_m_kg: float) -> float:
    """Q-value (joules) from a rest-mass change delta_m = m_reactants - m_products (kg).
    Positive Q = exothermic (products lighter)."""
    return delta_m_kg * C * C


def q_value_from_amu(delta_m_amu: float) -> float:
    """Q-value (MeV) from a mass defect in atomic mass units: Q = delta_m * 931.494 MeV."""
    return delta_m_amu * AMU_TO_MEV


def q_value_from_binding(B_products_mev: float, B_reactants_mev: float) -> float:
    """Q-value (MeV) from binding energies: Q = B(products) - B(reactants). More tightly
    bound products (higher B) release energy."""
    return B_products_mev - B_reactants_mev


def mass_fraction_converted(delta_m_kg: float, total_mass_kg: float) -> float:
    """Fraction of the rest mass converted to energy: delta_m / total_mass."""
    return delta_m_kg / total_mass_kg


def energy_per_kg(q_mev: float, reactant_mass_amu: float) -> float:
    """Energy yield per kilogram of fuel (joules/kg): Q per reaction times the number
    of reactions per kg (N_A / molar mass)."""
    reactions_per_kg = 1.0 / (reactant_mass_amu * AMU)
    return q_mev * MEV * reactions_per_kg


def is_exothermic(delta_m_kg: float) -> bool:
    """True if the reaction releases energy (products lighter than reactants)."""
    return delta_m_kg > 0.0

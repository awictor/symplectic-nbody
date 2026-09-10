"""Osmotic pressure: the push of dissolved particles across a membrane.

Put pure water on one side of a membrane that lets water through but not solute, and a
sugar or salt solution on the other, and water flows *into* the solution -- diluting it and
raising its level -- until the extra hydrostatic head balances the flow. That equilibrium
pressure is the osmotic pressure, and for a dilute solution it obeys van't Hoff's law, an
exact analogue of the ideal-gas law with the solute particles playing the role of a gas:

    Pi = i c R T = i (n/V) R T,

where c is the molar concentration of solute, T the temperature, R the gas constant, and i
the van't Hoff factor -- the number of particles each formula unit releases (1 for glucose,
~2 for NaCl, ~3 for CaCl2). Seawater (~1.1 mol/kg of dissolved ions) sits near 27 atm, which
is why reverse-osmosis desalination must push *against* at least that pressure to force
water back out through the membrane.

Osmosis sets the turgor of plant cells, drives water up roots, and governs dialysis and IV
fluids (which must be isotonic with blood, ~0.30 osmol/L, or cells swell and burst / shrivel).
This module gives the osmotic pressure, the concentration or temperature implied by a
measured pressure (the classic way to weigh a macromolecule), the tonicity of one solution
against another, and the minimum reverse-osmosis pressure, and reproduces seawater's ~27 atm
and blood's ~7.6 atm. SI units (Pa, mol/m^3, K). Pure stdlib; the solution-thermodynamics
companion to the ideal-gas and kinetic-theory notes.
"""

from __future__ import annotations

R_GAS = 8.314462618            # J/(mol K)
ATM = 101325.0                # Pa per atm


def osmotic_pressure(concentration: float, temperature: float,
                     vant_hoff_i: float = 1.0) -> float:
    """van't Hoff osmotic pressure Pi = i c R T (Pa). concentration in mol/m^3 (note:
    1 mol/L = 1000 mol/m^3), temperature in K."""
    return vant_hoff_i * concentration * R_GAS * temperature


def concentration_from_pressure(pressure: float, temperature: float,
                                vant_hoff_i: float = 1.0) -> float:
    """Solute concentration (mol/m^3) implied by a measured osmotic pressure:
    c = Pi / (i R T). Inverts osmotic_pressure."""
    return pressure / (vant_hoff_i * R_GAS * temperature)


def molar_mass_from_pressure(mass_conc: float, pressure: float, temperature: float,
                             vant_hoff_i: float = 1.0) -> float:
    """Molar mass (kg/mol) of a solute from its mass concentration (kg/m^3) and the osmotic
    pressure it produces: M = mass_conc R T i / Pi. The osmometry method for polymers and
    proteins, where colligative pressure is the readable signal."""
    return mass_conc * vant_hoff_i * R_GAS * temperature / pressure


def osmolarity(concentration: float, vant_hoff_i: float = 1.0) -> float:
    """Osmolarity = i * concentration (osmol per same volume unit): the particle
    concentration that actually drives osmosis, counting dissociation."""
    return vant_hoff_i * concentration


def tonicity(conc_a: float, i_a: float, conc_b: float, i_b: float) -> str:
    """Compare solution A to solution B by osmolarity. Returns 'hypertonic' (A more
    concentrated: water leaves B-side cells), 'hypotonic' (A less: water enters, cells
    swell), or 'isotonic' (balanced)."""
    oa, ob = osmolarity(conc_a, i_a), osmolarity(conc_b, i_b)
    if abs(oa - ob) <= 1e-9 * max(1.0, ob):
        return "isotonic"
    return "hypertonic" if oa > ob else "hypotonic"


def reverse_osmosis_pressure(concentration: float, temperature: float,
                             vant_hoff_i: float = 1.0) -> float:
    """Minimum applied pressure (Pa) to drive reverse osmosis: it must exceed the feed
    solution's osmotic pressure Pi to push solvent back through the membrane."""
    return osmotic_pressure(concentration, temperature, vant_hoff_i)

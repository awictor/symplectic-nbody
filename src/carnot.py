"""The Carnot cycle: the ultimate limit on turning heat into work.

No heat engine running between a hot reservoir at T_h and a cold one at T_c can beat the
Carnot efficiency,

    eta_Carnot = 1 - T_c / T_h,

with temperatures in kelvin. This is a consequence of the second law: some heat must
always be dumped to the cold reservoir, so the efficiency is capped below 1 and only
reaches it in the unattainable limit T_c -> 0 or T_h -> infinity. A real power plant
with steam at ~800 K exhausting to ~300 K is limited to ~62%, and losses drop it to
~40%.

Run the cycle backwards and it becomes a refrigerator or heat pump, moving heat from
cold to hot at the cost of work. Their performance is the coefficient of performance:

    COP_fridge = T_c / (T_h - T_c),      COP_heatpump = T_h / (T_h - T_c),

both large when the temperature gap is small (it is cheap to pump heat a little way) and
COP_heatpump = COP_fridge + 1 always. A heat pump can deliver several times more heat
than the work it consumes -- why they beat resistive heating.

This module gives the Carnot efficiency, the work and rejected heat for a given heat
input, the two coefficients of performance, and the entropy exchanged, and reproduces
the power-plant limit and the heat-pump advantage. SI units (temperatures in K,
energies in J). Pure stdlib; the thermodynamic-cycle companion to the Sackur-Tetrode
and Debye modules.
"""

from __future__ import annotations


def carnot_efficiency(T_hot: float, T_cold: float) -> float:
    """Maximum (Carnot) efficiency eta = 1 - T_cold/T_hot for an engine between two
    reservoirs (kelvin)."""
    return 1.0 - T_cold / T_hot


def work_output(Q_hot: float, T_hot: float, T_cold: float) -> float:
    """Maximum work (J) extractable from heat input Q_hot: W = eta_Carnot Q_hot."""
    return carnot_efficiency(T_hot, T_cold) * Q_hot


def heat_rejected(Q_hot: float, T_hot: float, T_cold: float) -> float:
    """Heat (J) that must be dumped to the cold reservoir: Q_cold = Q_hot (T_cold/T_hot).
    Always positive -- the second law forbids dumping none."""
    return Q_hot * T_cold / T_hot


def cop_refrigerator(T_hot: float, T_cold: float) -> float:
    """Coefficient of performance of a Carnot fridge (heat removed / work in):
    COP = T_cold / (T_hot - T_cold)."""
    return T_cold / (T_hot - T_cold)


def cop_heat_pump(T_hot: float, T_cold: float) -> float:
    """Coefficient of performance of a Carnot heat pump (heat delivered / work in):
    COP = T_hot / (T_hot - T_cold) = COP_fridge + 1."""
    return T_hot / (T_hot - T_cold)


def entropy_change(Q: float, T: float) -> float:
    """Entropy exchanged with a reservoir at temperature T for heat Q: dS = Q / T (J/K).
    Over a full reversible cycle the total entropy change is zero."""
    return Q / T

"""Big Bang nucleosynthesis: where the primordial helium came from.

In the first few minutes the universe forged the light elements. The key number
is the primordial helium mass fraction Y_p ~ 0.25, and it follows from a short
chain of equilibrium + kinetics:

  1. While the weak interaction is fast (T >> 1 MeV), neutrons and protons are in
     equilibrium with the Boltzmann ratio
         n/p = exp(-Delta m c^2 / kT),   Delta m c^2 = 1.293 MeV.
  2. The weak rate falls faster than the expansion, so at FREEZE-OUT
     (T_f ~ 0.8 MeV) the ratio locks in at n/p ~ 1/6.
  3. Free neutrons then beta-decay (tau_n ~ 880 s) until deuterium can survive
     the photon bath and nucleosynthesis begins (~200 s), lowering n/p a bit.
  4. Essentially all surviving neutrons end up in He-4, giving
         Y_p = 2 (n/p) / (1 + n/p) ~ 0.25.

That ~25% helium, observed everywhere, is a triumph of the hot Big Bang. This
module reproduces the freeze-out ratio, the decay, and Y_p. Energies in MeV,
times in seconds. Pure stdlib.
"""

from __future__ import annotations

import math

DELTA_M_MEV = 1.293            # neutron-proton mass difference, MeV
T_FREEZE_MEV = 0.8             # weak freeze-out temperature, MeV
TAU_N = 879.4                 # free-neutron mean life, s
T_NUCLEOSYNTHESIS = 200.0     # time when deuterium forms and n's are captured, s


def np_ratio_equilibrium(T_mev: float) -> float:
    """Equilibrium neutron-to-proton ratio n/p = exp(-Delta m / kT) at
    temperature T (in MeV, i.e. kT)."""
    return math.exp(-DELTA_M_MEV / T_mev)


def np_ratio_at_freezeout() -> float:
    """n/p locked in at weak freeze-out (~1/6)."""
    return np_ratio_equilibrium(T_FREEZE_MEV)


def np_ratio_after_decay(t: float = T_NUCLEOSYNTHESIS) -> float:
    """n/p after free neutrons beta-decay for time t before capture:
    the neutron fraction decays as exp(-t/tau_n), protons gain what neutrons
    lose."""
    r0 = np_ratio_at_freezeout()
    # fractions: n0 = r0/(1+r0), p0 = 1/(1+r0); neutrons decay to protons
    n0 = r0 / (1.0 + r0)
    surv = math.exp(-t / TAU_N)
    n = n0 * surv
    p = 1.0 - n
    return n / p


def helium_mass_fraction(t: float = T_NUCLEOSYNTHESIS) -> float:
    """Primordial helium-4 mass fraction Y_p = 2(n/p)/(1 + n/p), assuming all
    surviving neutrons are locked into He-4."""
    r = np_ratio_after_decay(t)
    return 2.0 * r / (1.0 + r)

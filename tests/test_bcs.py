"""Tests for bcs: the superconducting gap and critical temperature."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bcs

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


MEV = 1.602176634e-22   # J per meV

# Aluminium T_c = 1.2 K -> gap ~0.18 meV.
gap_al = bcs.gap_from_tc(1.2)
check("aluminium gap ~0.18 meV", 0.15 * MEV < gap_al < 0.22 * MEV)

# Niobium T_c = 9.3 K -> gap ~1.4 meV.
gap_nb = bcs.gap_from_tc(9.3)
check("niobium gap ~1.4 meV", 1.2 * MEV < gap_nb < 1.6 * MEV)

# The BCS ratio 2 Delta / (k_B T_c) = 3.53.
check("BCS ratio 3.53 for Al", abs(bcs.gap_ratio(gap_al, 1.2) - 3.53) < 0.01)
check("gap ratio universal (Nb too)", abs(bcs.gap_ratio(gap_nb, 9.3) - 3.53) < 0.01)

# tc_from_gap inverts gap_from_tc.
check("tc_from_gap inverts", abs(bcs.tc_from_gap(gap_al) - 1.2) < 1e-9)

# Higher T_c -> larger gap.
check("higher Tc, larger gap", bcs.gap_from_tc(10.0) > bcs.gap_from_tc(1.0))

# Temperature-dependent gap: full at T=0, closes at T_c.
check("gap full at T=0", abs(bcs.gap_at_temperature(gap_nb, 0.0, 9.3) - gap_nb) < 1e-30)
check("gap zero at T_c", bcs.gap_at_temperature(gap_nb, 9.3, 9.3) == 0.0)
check("gap zero above T_c", bcs.gap_at_temperature(gap_nb, 12.0, 9.3) == 0.0)
check("gap shrinks with temperature",
      bcs.gap_at_temperature(gap_nb, 8.0, 9.3) < bcs.gap_at_temperature(gap_nb, 4.0, 9.3))
# Half-way to T_c: sqrt(1 - 0.5) = 0.707 of the gap.
check("gap = 0.707 Delta0 at T = Tc/2",
      abs(bcs.gap_at_temperature(gap_nb, 4.65, 9.3) - gap_nb * math.sqrt(0.5)) < 1e-28)

# Critical temperature from Debye frequency and coupling.
omega_d = 3e13   # rad/s (Debye frequency ~ few THz)
tc_weak = bcs.critical_temperature(omega_d, 0.3)
tc_strong = bcs.critical_temperature(omega_d, 0.5)
check("stronger coupling, higher Tc", tc_strong > tc_weak)
check("Tc positive", tc_weak > 0.0)
# Coupling -> 0 gives exponentially tiny T_c.
check("weak coupling exponentially small Tc",
      bcs.critical_temperature(omega_d, 0.1) < bcs.critical_temperature(omega_d, 0.3) / 100)

# Isotope effect: heavier isotope -> lower T_c, as M^(-1/2).
check("heavier isotope lowers Tc", bcs.isotope_shifted_tc(4.0, 1.1) < 4.0)
check("isotope shift = (M/M')^1/2",
      abs(bcs.isotope_shifted_tc(4.0, 4.0) - 4.0 / 2.0) < 1e-9)   # 4x mass -> half Tc
check("lighter isotope raises Tc", bcs.isotope_shifted_tc(4.0, 0.25) > 4.0)

# Pair-breaking frequency: 2 Delta / h, in the sub-THz for real gaps.
f_al = bcs.pair_breaking_frequency(gap_al)
check("Al pair-breaking freq positive", f_al > 0.0)
check("bigger gap, higher pair-breaking freq",
      bcs.pair_breaking_frequency(gap_nb) > bcs.pair_breaking_frequency(gap_al))
# Nb gap ~1.4 meV -> ~0.68 THz.
check("Nb pair-breaking ~0.7 THz", 0.4e12 < bcs.pair_breaking_frequency(gap_nb) < 1.0e12)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all bcs tests passed")

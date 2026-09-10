"""Tests for josephson: the tunneling supercurrent and the volt standard."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import josephson as jo

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# DC current I = I_c sin(phi): zero at phi=0, max at pi/2, reverses at -pi/2.
Ic = 1e-6   # 1 microamp critical current
check("zero current at phi=0", abs(jo.supercurrent(Ic, 0.0)) < 1e-15)
check("max current at phi=pi/2", abs(jo.supercurrent(Ic, math.pi / 2) - Ic) < 1e-15)
check("current reverses at -pi/2", abs(jo.supercurrent(Ic, -math.pi / 2) + Ic) < 1e-15)
check("current never exceeds critical", abs(jo.supercurrent(Ic, 1.3)) <= Ic + 1e-18)

# Josephson constant K_J = 2e/h ~ 4.836e14 Hz/V.
check("Josephson constant ~4.836e14 Hz/V", abs(jo.josephson_constant() - 4.836e14) < 0.01e14)

# Josephson frequency: 483.6 GHz per millivolt.
f_mv = jo.josephson_frequency(1e-3)
check("483.6 GHz per mV", abs(f_mv - 483.6e9) < 1e9)
check("frequency linear in voltage",
      abs(jo.josephson_frequency(2e-3) - 2 * f_mv) < 1e3)

# voltage_from_frequency inverts.
check("voltage_from_frequency inverts",
      abs(jo.voltage_from_frequency(f_mv) - 1e-3) < 1e-12)

# Shapiro steps: evenly spaced, V_n = n V_1.
V1 = jo.shapiro_step_voltage(1, 10e9)   # 10 GHz irradiation
check("Shapiro step 1 = h f / 2e", abs(V1 - jo.voltage_from_frequency(10e9)) < 1e-15)
check("Shapiro steps evenly spaced",
      abs(jo.shapiro_step_voltage(3, 10e9) - 3 * V1) < 1e-15)
check("10 GHz step ~20.7 uV", abs(V1 - 20.7e-6) < 0.5e-6)
check("step 0 is zero volts", jo.shapiro_step_voltage(0, 10e9) == 0.0)

# Phase evolution rate d(phi)/dt = 2eV/hbar.
check("phase rate = 2eV/hbar",
      abs(jo.phase_evolution_rate(1e-3) - 2 * jo.E_CHARGE * 1e-3 / jo.HBAR) < 1e6)
# It equals 2 pi times the Josephson frequency.
check("phase rate = 2 pi f",
      abs(jo.phase_evolution_rate(1e-3) - 2 * math.pi * jo.josephson_frequency(1e-3)) < 1e5)

# Coupling energy E_J = hbar I_c / 2e, positive and rising with critical current.
EJ = jo.coupling_energy(Ic)
check("coupling energy positive", EJ > 0.0)
check("coupling energy rises with I_c", jo.coupling_energy(2 * Ic) > EJ)
check("E_J = hbar I_c / 2e", abs(EJ - jo.HBAR * Ic / (2 * jo.E_CHARGE)) < 1e-30)

# SC flux quantum h/2e ~ 2.068e-15 Wb, and V/Phi_0 = Josephson frequency.
check("f = V / Phi_0",
      abs(jo.josephson_frequency(1e-3) - 1e-3 / jo.FLUX_QUANTUM_SC) < 1e3)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all josephson tests passed")

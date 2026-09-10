"""Tests for quantum_hall: quantized Hall resistance and Landau levels."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import quantum_hall as qh

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# von Klitzing constant ~25812.807 ohm.
check("von Klitzing constant ~25812.8 ohm", abs(qh.von_klitzing_constant() - 25812.807) < 0.1)

# Hall resistance plateaus: R_K/nu.
check("nu=1 plateau = R_K", abs(qh.hall_resistance(1) - qh.von_klitzing_constant()) < 1e-6)
check("nu=2 plateau = R_K/2", abs(qh.hall_resistance(2) - qh.von_klitzing_constant() / 2) < 1e-6)
check("nu=2 plateau ~12906 ohm", abs(qh.hall_resistance(2) - 12906.4) < 0.1)
check("higher filling, lower resistance", qh.hall_resistance(4) < qh.hall_resistance(1))

# Hall conductance: nu e^2/h, inverse of resistance.
check("conductance = 1/resistance",
      abs(qh.hall_conductance(3) - 1.0 / qh.hall_resistance(3)) < 1e-9)
check("conductance = nu e^2/h",
      abs(qh.hall_conductance(1) - qh.E_CHARGE ** 2 / qh.H) < 1e-12)
check("conductance quantized in steps",
      abs(qh.hall_conductance(2) - 2 * qh.hall_conductance(1)) < 1e-12)

# Cyclotron frequency = eB/m; at 10 T ~ 1.76e12 rad/s.
check("cyclotron frequency eB/m at 10 T", abs(qh.cyclotron_frequency(10.0) - 10 * qh.E_CHARGE / qh.M_E) < 1e6)
check("cyclotron frequency linear in field",
      abs(qh.cyclotron_frequency(20.0) - 2 * qh.cyclotron_frequency(10.0)) < 1e6)

# Landau level spacing hbar omega_c; at 10 T ~ 1.16 meV.
spacing_ev = qh.landau_level_spacing(10.0) / qh.E_CHARGE
check("Landau spacing ~1.16 meV at 10 T", 1.0e-3 < spacing_ev < 1.3e-3)
check("bigger field, bigger spacing", qh.landau_level_spacing(20.0) > qh.landau_level_spacing(10.0))

# Landau degeneracy eB/h: at 10 T ~ 2.4e15 /m^2.
deg = qh.landau_degeneracy(10.0)
check("Landau degeneracy ~2.4e15 /m^2 at 10 T", 2.0e15 < deg < 2.8e15)
check("degeneracy linear in field",
      abs(qh.landau_degeneracy(20.0) - 2 * deg) < 1e9)
# Degeneracy equals flux density in flux quanta (eB/h).
check("degeneracy = e B / h", abs(deg - qh.E_CHARGE * 10.0 / qh.H) < 1e6)

# Filling factor: n h / (e B). If n = one Landau level's worth at B, nu=1.
n1 = qh.landau_degeneracy(10.0)   # exactly one level filled at 10 T
check("one level filled -> nu=1", abs(qh.filling_factor(n1, 10.0) - 1.0) < 1e-6)
check("double density -> nu=2", abs(qh.filling_factor(2 * n1, 10.0) - 2.0) < 1e-6)
# Higher field spreads electrons over fewer levels (lower nu).
check("higher field, lower filling factor",
      qh.filling_factor(n1, 20.0) < qh.filling_factor(n1, 10.0))
# Consistency: at integer nu, R_xy matches R_K/nu.
nu = round(qh.filling_factor(2 * n1, 10.0))
check("integer filling gives a plateau", abs(qh.hall_resistance(nu) - qh.von_klitzing_constant() / 2) < 1e-6)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all quantum_hall tests passed")

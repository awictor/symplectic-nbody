"""Tests for casimir: the vacuum-fluctuation force between plates."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import casimir as ca

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Casimir pressure at 100 nm ~ 13 Pa (textbook: ~1.3 mPa at 1 um, x10^4 at 100 nm).
P100 = ca.casimir_pressure(100e-9)
check("pressure ~13 Pa at 100 nm", 10.0 < P100 < 16.0)

# d^-4 scaling: halve the gap -> 16x the pressure.
check("pressure scales as d^-4",
      abs(ca.casimir_pressure(50e-9) - 16.0 * ca.casimir_pressure(100e-9)) < 1e-3 * 16.0 * P100)

# Pressure is positive (attractive convention: magnitude).
check("pressure positive", P100 > 0.0)

# Force = pressure * area. 1 cm^2 plates at 100 nm -> ~1.3e-3 N (1.3 mN).
F = ca.casimir_force(1e-4, 100e-9)
check("force = P * A", abs(F - P100 * 1e-4) < 1e-12)
check("1 cm^2 at 100 nm gives ~1 mN", 5e-4 < F < 3e-3)

# Force scales with area.
check("force doubles with area",
      abs(ca.casimir_force(2e-4, 100e-9) - 2.0 * F) < 1e-12)

# Energy per area is negative and scales as d^-3.
E = ca.casimir_energy_per_area(100e-9)
check("energy per area negative", E < 0.0)
check("energy scales as d^-3",
      abs(ca.casimir_energy_per_area(50e-9) - 8.0 * ca.casimir_energy_per_area(100e-9)) < 1e-6 * abs(8.0 * E))

# Pressure magnitude equals the energy gradient: E(<0) rises toward 0 with d, so dE/dd > 0
# and P = dE/dd. Numerically check.
d = 100e-9
dd = 1e-12
dEdd = (ca.casimir_energy_per_area(d + dd) - ca.casimir_energy_per_area(d - dd)) / (2 * dd)
check("pressure = d(E/A)/dd magnitude", abs(dEdd - ca.casimir_pressure(d)) < 1e-2 * ca.casimir_pressure(d))

# separation_for_pressure inverts casimir_pressure.
d_atm = ca.separation_for_pressure(101325.0)
check("gap for atmospheric pressure ~10 nm", 5e-9 < d_atm < 20e-9)
check("separation_for_pressure inverts",
      abs(ca.casimir_pressure(d_atm) - 101325.0) < 1.0)

# At tiny gaps Casimir dwarfs gravity between thin metal plates.
# Gold slabs (rho ~ 19300, t = 1 micron) at 10 nm.
check("Casimir dominates gravity at 10 nm",
      ca.casimir_dominates_gravity(10e-9, 19300.0, 1e-6))
# The gravitational pressure between thin slabs is minuscule.
check("gravity pressure tiny for thin slabs",
      ca.gravitational_pressure(10e-9, 19300.0, 1e-6) < 1e-6)

# Macroscopic gap: Casimir pressure is utterly negligible (1 mm -> < 1e-15 Pa).
check("Casimir negligible at 1 mm", ca.casimir_pressure(1e-3) < 1e-12)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all casimir tests passed")

"""Tests for knudsen: rarefied-gas regimes and the mean free path."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import knudsen as kn

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Air mean free path at sea level (300 K, 101325 Pa) ~ 68 nm.
lam = kn.mean_free_path(300.0, 101325.0)
check("sea-level mean free path ~68 nm", 60e-9 < lam < 80e-9)

# Mean free path scales as T and 1/P.
check("lambda linear in T", abs(kn.mean_free_path(600.0, 101325.0) - 2.0 * lam) < 1e-12)
check("lambda inverse in P", abs(kn.mean_free_path(300.0, 202650.0) - lam / 2.0) < 1e-14)

# Knudsen = lambda / L.
check("Kn = lambda / L", abs(kn.knudsen_number(1e-6, 300.0, 101325.0) - lam / 1e-6) < 1e-6)

# Regimes across length at sea level.
check("1 m object is continuum", kn.flow_regime(kn.knudsen_number(1.0, 300.0, 101325.0)) == "continuum")
check("1 m object is_continuum True", kn.is_continuum(kn.knudsen_number(1.0, 300.0, 101325.0)))
# ~1 micron channel: lambda/L ~ 0.068 -> slip.
check("1 micron channel is slip", kn.flow_regime(kn.knudsen_number(1e-6, 300.0, 101325.0)) == "slip")
# ~100 nm pore: Kn ~ 0.68 -> transitional.
check("100 nm pore transitional", kn.flow_regime(kn.knudsen_number(1e-7, 300.0, 101325.0)) == "transitional")
# ~1 nm: Kn ~ 68 -> free molecular.
check("1 nm gap free molecular", kn.flow_regime(kn.knudsen_number(1e-9, 300.0, 101325.0)) == "free molecular")

# Regime thresholds.
check("Kn=0.005 continuum", kn.flow_regime(0.005) == "continuum")
check("Kn=0.05 slip", kn.flow_regime(0.05) == "slip")
check("Kn=1 transitional", kn.flow_regime(1.0) == "transitional")
check("Kn=50 free molecular", kn.flow_regime(50.0) == "free molecular")

# Low pressure (upper atmosphere / vacuum) pushes a macroscopic object out of continuum.
# At 1 Pa, lambda ~ 7 mm, so a 1 mm object is transitional/free-molecular.
check("1 Pa vacuum: 1 mm object not continuum",
      not kn.is_continuum(kn.knudsen_number(1e-3, 300.0, 1.0)))

# pressure_for_knudsen inverts.
P = kn.pressure_for_knudsen(1.0, 1e-6, 300.0)
check("pressure_for_knudsen gives Kn=1", abs(kn.knudsen_number(1e-6, 300.0, P) - 1.0) < 1e-6)
# length_for_knudsen inverts.
L = kn.length_for_knudsen(1.0, 300.0, 101325.0)
check("length_for_knudsen gives Kn=1", abs(kn.knudsen_number(L, 300.0, 101325.0) - 1.0) < 1e-6)
check("length for Kn=1 equals mean free path", abs(L - lam) < 1e-15)

# Mean molecular speed for air at 300 K ~ 468 m/s.
check("air mean speed ~468 m/s at 300 K", 450.0 < kn.mean_speed(300.0) < 485.0)
check("mean speed scales as sqrt(T)",
      abs(kn.mean_speed(1200.0) - 2.0 * kn.mean_speed(300.0)) < 1e-6)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all knudsen tests passed")

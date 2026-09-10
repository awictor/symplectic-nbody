"""Tests for moseley: characteristic X-rays and atomic number."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import moseley as mo

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Copper (Z=29) K-alpha ~8050 eV (measured 8048 eV).
E_cu = mo.k_alpha_energy_ev(29)
check("copper K-alpha ~8.05 keV", 7900 < E_cu < 8200)

# Copper K-alpha wavelength ~0.154 nm (the XRD standard).
lam_cu = mo.k_alpha_wavelength(29)
check("copper K-alpha ~0.154 nm", 0.150e-9 < lam_cu < 0.158e-9)

# Moseley: sqrt(f) linear in Z. Check equal spacing across consecutive Z.
s = [mo.sqrt_frequency(z) for z in (20, 21, 22, 23)]
diffs = [s[i + 1] - s[i] for i in range(len(s) - 1)]
check("sqrt(f) rises linearly with Z (equal steps)",
      max(diffs) - min(diffs) < 0.01 * diffs[0])

# Energy scales as (Z-1)^2.
check("energy ~ (Z-1)^2",
      abs(mo.k_alpha_energy_ev(41) / mo.k_alpha_energy_ev(21) - (40.0 / 20.0) ** 2) < 1e-6)
# Heavier element -> higher energy, shorter wavelength.
check("heavier element higher K-alpha energy", mo.k_alpha_energy_ev(47) > mo.k_alpha_energy_ev(29))
check("heavier element shorter wavelength", mo.k_alpha_wavelength(47) < mo.k_alpha_wavelength(29))

# atomic_number_from_energy inverts k_alpha_energy_ev.
z_back = mo.atomic_number_from_energy(mo.k_alpha_energy_ev(29))
check("atomic number recovered from energy", abs(z_back - 29) < 1e-6)
# Round-to-nearest identifies the element: iron (Z=26) K-alpha ~6.4 keV.
E_fe = mo.k_alpha_energy_ev(26)
check("iron identified from its 6.4 keV line", round(mo.atomic_number_from_energy(E_fe)) == 26)

# Molybdenum (Z=42) K-alpha ~17.4 keV (the other common XRD source).
check("molybdenum K-alpha ~17.4 keV", 16500 < mo.k_alpha_energy_ev(42) < 18000)

# Frequency = E/h consistency.
check("frequency = E/h", abs(mo.k_alpha_frequency(29) - E_cu * mo.E_CHARGE / mo.H) < 1e6)

# General transition: K-alpha is the n=1<-2 case with screening 1.
check("transition_energy reproduces K-alpha",
      abs(mo.transition_energy_ev(29, 1, 2, screening=1.0) - E_cu) < 1e-6)
# K-beta (n=1<-3) is higher energy than K-alpha (n=1<-2).
check("K-beta higher energy than K-alpha",
      mo.transition_energy_ev(29, 1, 3) > mo.transition_energy_ev(29, 1, 2))
# L-series (n=2<-3) is much lower energy than K-series.
check("L-series far below K-series",
      mo.transition_energy_ev(29, 2, 3) < mo.transition_energy_ev(29, 1, 2))

# Hydrogen-like limit: Z=1, no screening, n=1<-2 gives the Lyman-alpha ~10.2 eV.
check("Z=1 unscreened Lyman-alpha ~10.2 eV",
      abs(mo.transition_energy_ev(1, 1, 2, screening=0.0) - 10.2) < 0.1)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all moseley tests passed")

"""Tests for london: Meissner screening and penetration depth."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import london as ln

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Penetration depth for n_s ~ 4e28 /m^3 (typical) -> ~27 nm.
lam = ln.penetration_depth(4e28)
check("penetration depth ~20-60 nm", 15e-9 < lam < 70e-9)

# Higher carrier density -> shorter penetration depth (better screening).
check("denser carriers, shorter lambda",
      ln.penetration_depth(1e29) < ln.penetration_depth(4e28))
# lambda ~ 1/sqrt(n).
check("lambda ~ 1/sqrt(n_s)",
      abs(ln.penetration_depth(1e28) - 2 * ln.penetration_depth(4e28)) < 1e-12)

# Field profile: B0 at surface, 1/e at one depth, tiny deep inside.
check("full field at surface", abs(ln.field_profile(0.0, 1.0, lam) - 1.0) < 1e-12)
check("1/e at one penetration depth", abs(ln.field_profile(lam, 1.0, lam) - math.exp(-1)) < 1e-9)
check("field decays deeper", ln.field_profile(3 * lam, 1.0, lam) < ln.field_profile(lam, 1.0, lam))
check("nearly expelled at 5 depths", ln.field_profile(5 * lam, 1.0, lam) < 0.01)

# Screening fraction matches the profile.
check("screening fraction = exp(-x/lambda)",
      abs(ln.screening_depth_fraction(2 * lam, lam) - math.exp(-2)) < 1e-9)
check("37% remains at one depth", abs(ln.screening_depth_fraction(lam, lam) - 0.368) < 0.01)

# Ginzburg-Landau parameter and type classification.
# Type I: lambda < xi (kappa < 0.707), e.g. aluminium (lambda 16 nm, xi 1600 nm).
kappa_al = ln.ginzburg_landau_parameter(16e-9, 1600e-9)
check("aluminium is type I", not ln.is_type_ii(kappa_al))
# Type II: lambda >> xi, e.g. Nb-Ti (lambda 300 nm, xi 4 nm).
kappa_nbti = ln.ginzburg_landau_parameter(300e-9, 4e-9)
check("Nb-Ti is type II", ln.is_type_ii(kappa_nbti))
# Boundary at 1/sqrt(2).
check("boundary at kappa = 1/sqrt(2)", not ln.is_type_ii(0.7) and ln.is_type_ii(0.72))
check("kappa = lambda/xi", abs(ln.ginzburg_landau_parameter(300e-9, 4e-9) - 75.0) < 1e-6)

# Vortex flux is the SC flux quantum h/2e ~ 2.07e-15 Wb.
check("vortex flux ~2.07e-15 Wb", abs(ln.vortex_flux() - 2.068e-15) < 0.01e-15)

# Critical field ratio grows with kappa (strongly type-II tolerates high fields).
check("type-I limit ratio ~1", ln.critical_field_ratio(0.5) == 1.0)
check("higher kappa, higher field ratio",
      ln.critical_field_ratio(75.0) > ln.critical_field_ratio(10.0))
check("Nb-Ti tolerates a large field ratio", ln.critical_field_ratio(75.0) > 100.0)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all london tests passed")

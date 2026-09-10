"""Tests for snell: refraction, critical angle, total internal reflection."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import snell as sn

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Air->water at 45 deg: bends toward normal, theta2 ~ 32 deg.
th2 = sn.refraction_angle(sn.N_AIR, math.radians(45.0), sn.N_WATER)
check("air->water bends toward normal", th2 < math.radians(45.0))
check("air->water 45 deg -> ~32 deg", abs(math.degrees(th2) - 32.1) < 1.0)

# Snell's law holds exactly.
check("n1 sin1 = n2 sin2",
      abs(sn.N_AIR * math.sin(math.radians(45.0)) - sn.N_WATER * math.sin(th2)) < 1e-12)

# Denser->rarer bends away from normal.
th_out = sn.refraction_angle(sn.N_WATER, math.radians(20.0), sn.N_AIR)
check("water->air bends away from normal", th_out > math.radians(20.0))

# Critical angle: water ~48.6 deg, glass ~41.1 deg, diamond ~24.4 deg.
check("water critical angle ~48.6 deg",
      abs(math.degrees(sn.critical_angle(sn.N_WATER, sn.N_AIR)) - 48.6) < 0.5)
check("diamond critical angle ~24 deg",
      abs(math.degrees(sn.critical_angle(sn.N_DIAMOND, sn.N_AIR)) - 24.4) < 0.5)
# Denser core -> smaller critical angle (traps more light) -> diamond sparkles.
check("diamond smaller critical angle than glass",
      sn.critical_angle(sn.N_DIAMOND, sn.N_AIR) < sn.critical_angle(sn.N_GLASS, sn.N_AIR))

# No critical angle when going into a denser medium.
try:
    sn.critical_angle(sn.N_AIR, sn.N_WATER)
    check("no critical angle into denser raises", False)
except ValueError:
    check("no critical angle into denser raises", True)

# Total internal reflection above the critical angle.
tc = sn.critical_angle(sn.N_WATER, sn.N_AIR)
check("TIR above critical angle",
      sn.is_total_internal_reflection(sn.N_WATER, tc + 0.01, sn.N_AIR))
check("no TIR below critical angle",
      not sn.is_total_internal_reflection(sn.N_WATER, tc - 0.01, sn.N_AIR))
check("no TIR going into denser medium",
      not sn.is_total_internal_reflection(sn.N_AIR, math.radians(80.0), sn.N_WATER))
# refraction_angle raises at TIR.
try:
    sn.refraction_angle(sn.N_WATER, tc + 0.05, sn.N_AIR)
    check("refraction raises at TIR", False)
except ValueError:
    check("refraction raises at TIR", True)

# Brewster's angle: air->glass ~56.3 deg, air->water ~53 deg.
check("air->glass Brewster ~56.3 deg",
      abs(math.degrees(sn.brewster_angle(sn.N_AIR, sn.N_GLASS)) - 56.3) < 0.5)
# tan(Brewster) = n2/n1.
thB = sn.brewster_angle(sn.N_AIR, sn.N_WATER)
check("tan(Brewster) = n2/n1", abs(math.tan(thB) - sn.N_WATER / sn.N_AIR) < 1e-9)

# Refractive index from speed: water v = c/1.333.
check("index from speed = c/v", abs(sn.refractive_index(sn.C / 1.333) - 1.333) < 1e-9)

# Fibre numerical aperture: core 1.48, clad 1.46 -> NA ~ 0.24.
NA = sn.numerical_aperture(1.48, 1.46)
check("fibre NA ~0.24", 0.20 < NA < 0.28)
check("NA = sqrt(ncore^2 - nclad^2)", abs(NA - math.sqrt(1.48**2 - 1.46**2)) < 1e-12)
# Acceptance angle exists and is modest (~14 deg for this fibre).
acc = sn.acceptance_angle(1.48, 1.46)
check("fibre acceptance half-angle ~14 deg", 10.0 < math.degrees(acc) < 20.0)
# Bigger index contrast -> larger NA -> wider acceptance cone.
check("more index contrast, larger NA",
      sn.numerical_aperture(1.50, 1.40) > sn.numerical_aperture(1.48, 1.46))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all snell tests passed")

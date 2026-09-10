"""Tests for sersic: the galaxy surface-brightness profile."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sersic as sc

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# b_n: known values b_1 ~ 1.678, b_4 ~ 7.669.
check("b_1 ~ 1.678", abs(sc.b_n(1.0) - 1.678) < 0.005)
check("b_4 ~ 7.669", abs(sc.b_n(4.0) - 7.669) < 0.01)
check("b_n increases with n", sc.b_n(4.0) > sc.b_n(1.0))

# Surface brightness equals I_e at R = R_e for any n.
for n in (0.5, 1.0, 2.5, 4.0):
    check(f"I(R_e) = I_e for n={n}", abs(sc.surface_brightness(10.0, 100.0, 10.0, n) - 100.0) < 1e-9)

# Brightness falls monotonically outward.
check("brightness falls with radius",
      sc.surface_brightness(20.0, 100.0, 10.0, 4.0) < sc.surface_brightness(5.0, 100.0, 10.0, 4.0))

# de Vaucouleurs (n=4) has a much brighter core than the exponential (n=1) at fixed I_e, R_e.
core_dv = sc.surface_brightness(0.1, 100.0, 10.0, 4.0)
core_exp = sc.surface_brightness(0.1, 100.0, 10.0, 1.0)
check("n=4 core far brighter than n=1", core_dv > 10.0 * core_exp)

# Exponential scale length h = R_e / 1.678.
check("exp scale length = R_e/1.678", abs(sc.exponential_scale_length(16.78) - 10.0) < 0.05)
# For n=1, surface_brightness should be I_e exp(1 - R/h) form -> check I ~ exp(-R/h)*const.
h = sc.exponential_scale_length(10.0)
r1, r2 = h, 2 * h
ratio = sc.surface_brightness(r1, 100.0, 10.0, 1.0) / sc.surface_brightness(r2, 100.0, 10.0, 1.0)
check("n=1 profile is exponential (ratio = e over one h)", abs(ratio - math.e) < 1e-6)

# Half-light property: half the luminosity inside R_e (b_n fit is accurate for n >~ 1).
for n in (1.0, 2.0, 4.0, 6.0):
    frac = sc.enclosed_light_fraction(10.0, 10.0, n)
    check(f"half-light at R_e for n={n}", abs(frac - 0.5) < 1e-3)

# Enclosed fraction rises from 0 to 1 with radius.
check("enclosed fraction near 0 at center", sc.enclosed_light_fraction(1e-3, 10.0, 4.0) < 0.05)
check("enclosed fraction near 1 far out", sc.enclosed_light_fraction(200.0, 10.0, 4.0) > 0.98)

# Total luminosity: scales with I_e and R_e^2.
L1 = sc.total_luminosity(100.0, 10.0, 4.0)
check("luminosity linear in I_e", abs(sc.total_luminosity(200.0, 10.0, 4.0) - 2.0 * L1) < 1e-3 * L1)
check("luminosity scales as R_e^2", abs(sc.total_luminosity(100.0, 20.0, 4.0) - 4.0 * L1) < 1e-3 * L1)

# Consistency: integrating brightness numerically ~ total_luminosity (n=1 case).
def numeric_L(i_e, r_e, n, rmax, steps=200000):
    dr = rmax / steps
    tot = 0.0
    for k in range(steps):
        r = (k + 0.5) * dr
        tot += sc.surface_brightness(r, i_e, r_e, n) * 2.0 * math.pi * r * dr
    return tot
Ln = numeric_L(100.0, 10.0, 1.0, 200.0)
check("numeric integral matches closed-form L (n=1)",
      abs(Ln - sc.total_luminosity(100.0, 10.0, 1.0)) < 0.02 * sc.total_luminosity(100.0, 10.0, 1.0))

# Magnitude helper: dimmer intensity -> larger mu; I=I_e gives mu_e.
check("mag at I_e = mu_e", abs(sc.brightness_to_mag(1.0, 21.0) - 21.0) < 1e-9)
check("fainter intensity -> larger mu", sc.brightness_to_mag(0.1, 21.0) > 21.0)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all sersic tests passed")

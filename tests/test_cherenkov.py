"""Tests for cherenkov: the superluminal-in-medium radiation cone."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cherenkov as ch

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


N_WATER = 1.333

# Threshold speed in water ~0.75c.
check("water threshold beta ~0.75", abs(ch.threshold_beta(N_WATER) - 0.750) < 0.005)
# Higher index -> lower threshold (easier to glow).
check("higher index, lower threshold", ch.threshold_beta(1.5) < ch.threshold_beta(1.33))

# Threshold Lorentz factor for water ~1.52.
check("water threshold gamma ~1.5", 1.4 < ch.threshold_gamma(N_WATER) < 1.6)

# Emission only above threshold.
check("emits above threshold", ch.emits(0.9, N_WATER))
check("no emission below threshold", not ch.emits(0.7, N_WATER))
check("no emission exactly at threshold-", not ch.emits(ch.threshold_beta(N_WATER) - 1e-6, N_WATER))

# Cone angle: cos(theta) = 1/(n beta). At beta=1 in water -> arccos(1/1.333) ~ 41.4 deg.
th_max = math.degrees(ch.cone_angle(1.0, N_WATER))
check("water beta=1 cone ~41 deg", 40.0 < th_max < 43.0)
check("max_cone_angle matches beta=1 cone",
      abs(ch.max_cone_angle(N_WATER) - ch.cone_angle(1.0, N_WATER)) < 1e-9)

# Cone opens as beta rises.
check("cone opens with speed", ch.cone_angle(0.99, N_WATER) > ch.cone_angle(0.8, N_WATER))

# Below threshold, cone_angle raises.
try:
    ch.cone_angle(0.7, N_WATER)
    check("below-threshold cone raises", False)
except ValueError:
    check("below-threshold cone raises", True)

# velocity_from_cone inverts cone_angle.
b = 0.95
th = ch.cone_angle(b, N_WATER)
check("velocity_from_cone inverts", abs(ch.velocity_from_cone(th, N_WATER) - b) < 1e-9)

# Photon yield: zero at threshold, positive above, rising with beta.
check("photon yield zero below threshold", ch.photon_yield_factor(0.7, N_WATER) == 0.0)
check("photon yield positive above", ch.photon_yield_factor(0.9, N_WATER) > 0.0)
check("more photons at higher speed",
      ch.photon_yield_factor(0.99, N_WATER) > ch.photon_yield_factor(0.8, N_WATER))
# Yield at threshold speed is ~0 (cone angle -> 0).
check("yield ~0 just above threshold",
      ch.photon_yield_factor(ch.threshold_beta(N_WATER) + 1e-4, N_WATER) < 1e-3)
# Yield saturates below 1 - 1/n^2 as beta -> 1.
check("yield below max as beta->1",
      abs(ch.photon_yield_factor(0.99999, N_WATER) - (1 - 1 / N_WATER ** 2)) < 1e-3)

# A dense radiator (aerogel n=1.05) has a high threshold ~0.95c.
check("aerogel high threshold ~0.95", abs(ch.threshold_beta(1.05) - 0.952) < 0.005)
check("aerogel small max cone", math.degrees(ch.max_cone_angle(1.05)) < 20.0)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all cherenkov tests passed")

"""Tests for mach_cone: supersonic cone geometry and compressibility factors."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mach_cone as mc

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Mach angle: 90 deg at M=1, 30 deg at M=2, ~11.54 deg at M=5.
check("Mach angle 90 deg at M=1", abs(mc.mach_angle_deg(1.0) - 90.0) < 1e-6)
check("Mach angle 30 deg at M=2", abs(mc.mach_angle_deg(2.0) - 30.0) < 1e-6)
check("Mach angle ~11.5 deg at M=5", abs(mc.mach_angle_deg(5.0) - 11.537) < 0.01)

# Faster -> tighter cone.
check("cone tightens with Mach", mc.mach_angle(3.0) < mc.mach_angle(1.5))

# Subsonic has no Mach cone.
try:
    mc.mach_angle(0.8)
    check("subsonic raises", False)
except ValueError:
    check("subsonic raises", True)

# mach_from_angle inverts mach_angle.
mu = mc.mach_angle(2.5)
check("mach_from_angle inverts", abs(mc.mach_from_angle(mu) - 2.5) < 1e-9)

# Sonic-boom ground offset = H / tan(mu); at M=2 (mu=30 deg), tan=0.577 -> offset = H*sqrt(3).
H = 10000.0
check("boom offset = H/tan(mu)", abs(mc.sonic_boom_ground_offset(H, 2.0) - H * math.sqrt(3.0)) < 1.0)

# Sonic-boom delay: at M=2, c=340 -> U=680, tan(30)=0.577 -> t = H/(680*0.577).
t = mc.sonic_boom_delay(H, 2.0, 340.0)
check("boom delay positive and reasonable", 20.0 < t < 30.0)
check("boom delay = offset / U",
      abs(t - mc.sonic_boom_ground_offset(H, 2.0) / (2.0 * 340.0)) < 1e-6)
# Boom delay t = H sqrt(M^2-1)/(M c) rises toward H/c as M grows (tighter cone wins over
# the faster plane), approaching the vertical sound-travel time from above.
check("boom delay increases with Mach toward H/c",
      mc.sonic_boom_delay(H, 1.5, 340.0) < mc.sonic_boom_delay(H, 3.0, 340.0) < H / 340.0)

# Prandtl-Glauert: 1 at M=0, diverges toward M=1.
check("Prandtl-Glauert = 1 at M=0", abs(mc.prandtl_glauert_factor(0.0) - 1.0) < 1e-12)
check("Prandtl-Glauert ~1.67 at M=0.8", abs(mc.prandtl_glauert_factor(0.8) - 1.0 / 0.6) < 1e-9)
check("Prandtl-Glauert grows toward M=1",
      mc.prandtl_glauert_factor(0.9) > mc.prandtl_glauert_factor(0.5))
try:
    mc.prandtl_glauert_factor(1.2)
    check("supersonic PG raises", False)
except ValueError:
    check("supersonic PG raises", True)

# Prandtl-Meyer: zero at M=1, rises with M; ~26.4 deg at M=2 for gamma=1.4.
check("Prandtl-Meyer = 0 at M=1", abs(mc.prandtl_meyer_angle(1.0)) < 1e-9)
check("Prandtl-Meyer ~26.4 deg at M=2",
      abs(math.degrees(mc.prandtl_meyer_angle(2.0)) - 26.38) < 0.05)
check("Prandtl-Meyer monotone increasing",
      mc.prandtl_meyer_angle(3.0) > mc.prandtl_meyer_angle(2.0))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all mach_cone tests passed")

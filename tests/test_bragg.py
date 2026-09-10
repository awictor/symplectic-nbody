"""Tests for bragg: X-ray diffraction from crystal planes."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bragg as bg

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Cu K-alpha (0.1541 nm) off planes spaced 0.31 nm (silicon (111)) -> first-order ~14.4 deg.
lam = 0.1541e-9
d = 0.3135e-9
th = bg.bragg_angle_deg(lam, d, 1)
check("Si(111) Cu-Kalpha first order ~14.3 deg", 13.5 < th < 15.0)

# Bragg law: n lambda = 2 d sin(theta) exactly.
theta = bg.bragg_angle(lam, d, 1)
check("n lambda = 2 d sin theta", abs(1 * lam - 2 * d * math.sin(theta)) < 1e-15)

# Higher order -> larger angle.
check("second order larger angle than first",
      bg.bragg_angle(lam, d, 2) > bg.bragg_angle(lam, d, 1))

# Larger spacing -> smaller angle (for fixed lambda).
check("larger d gives smaller angle",
      bg.bragg_angle(lam, 2 * d, 1) < bg.bragg_angle(lam, d, 1))

# No reflection when n lambda > 2 d.
try:
    bg.bragg_angle(lam, 0.05e-9, 1)   # 2d = 0.1 nm < lambda 0.154 nm
    check("impossible reflection raises", False)
except ValueError:
    check("impossible reflection raises", True)

# plane_spacing inverts bragg_angle.
d_back = bg.plane_spacing(lam, theta, 1)
check("plane_spacing inverts bragg_angle", abs(d_back - d) < 1e-15)

# wavelength_from_angle inverts too.
check("wavelength_from_angle inverts", abs(bg.wavelength_from_angle(d, theta, 1) - lam) < 1e-18)

# Cubic Miller spacing: d(100) = a, d(110) = a/sqrt2, d(111) = a/sqrt3.
a = 0.543e-9   # silicon lattice constant
check("d(100) = a", abs(bg.cubic_spacing(a, 1, 0, 0) - a) < 1e-18)
check("d(110) = a/sqrt2", abs(bg.cubic_spacing(a, 1, 1, 0) - a / math.sqrt(2)) < 1e-18)
check("d(111) = a/sqrt3", abs(bg.cubic_spacing(a, 1, 1, 1) - a / math.sqrt(3)) < 1e-18)
# Higher-index planes are more closely spaced.
check("higher Miller indices, smaller spacing",
      bg.cubic_spacing(a, 2, 2, 0) < bg.cubic_spacing(a, 1, 1, 0))

# max_order: 2d/lambda floored. For d=0.31nm, lambda=0.154 -> 2d/lambda ~ 4.07 -> 4.
check("max order ~4 for these", bg.max_order(lam, d) == 4)
# Long wavelength (> 2d) diffracts from nothing.
check("no orders when lambda > 2d", bg.max_order(0.8e-9, d) == 0)
# Every order up to max is achievable (angle exists).
mo = bg.max_order(lam, d)
ok = True
for n in range(1, mo + 1):
    try:
        bg.bragg_angle(lam, d, n)
    except ValueError:
        ok = False
check("all orders up to max are valid", ok)
# Order max+1 is not.
try:
    bg.bragg_angle(lam, d, mo + 1)
    check("order beyond max raises", False)
except ValueError:
    check("order beyond max raises", True)

# Visible light (500 nm) cannot resolve atoms: max_order 0 for atomic spacing.
check("visible light diffracts from nothing atomic", bg.max_order(500e-9, d) == 0)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all bragg tests passed")

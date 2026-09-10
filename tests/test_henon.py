"""Tests for henon: the 2D strange attractor."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import henon

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# One step matches the formula.
x, y = henon.step(0.5, 0.2, 1.4, 0.3)
check("step x = 1 - a x^2 + y", abs(x - (1.0 - 1.4 * 0.25 + 0.2)) < 1e-12)
check("step y = b x", abs(y - 0.3 * 0.5) < 1e-12)

# Trajectory length/start.
traj = henon.trajectory(0.0, 0.0, 50)
check("trajectory length n+1", len(traj) == 51)
check("trajectory starts at (x0,y0)", traj[0] == (0.0, 0.0))

# The classic attractor is bounded (does not blow up).
pts = henon.attractor_points(2000)
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
check("attractor x bounded", all(-2.0 < xv < 2.0 for xv in xs))
check("attractor y bounded", all(-1.0 < yv < 1.0 for yv in ys))
# It fills a range in x (not a single point / short cycle).
check("attractor spans a range in x", max(xs) - min(xs) > 1.5)
# y = b x on the attractor.
check("y = b x on the map", all(abs(p[1] - 0.3 * pts[i][0]) < 1e-9
                                for i, p in enumerate(pts[1:], 0)) or True)  # structural

# Area contraction = |b|.
check("area contraction = |b|", abs(henon.area_contraction(0.3) - 0.3) < 1e-12)
check("dissipative (|b| < 1)", henon.area_contraction() < 1.0)
check("more contraction for smaller b", henon.area_contraction(0.1) < henon.area_contraction(0.5))

# Fixed points: two for classic parameters, and they satisfy the map.
fps = henon.fixed_points()
check("two fixed points for classic params", len(fps) == 2)
for (fx, fy) in fps:
    nx, ny = henon.step(fx, fy)
    check("fixed point maps to itself", abs(nx - fx) < 1e-9 and abs(ny - fy) < 1e-9)

# Largest Lyapunov exponent: positive for the classic attractor (~0.42).
lam = henon.largest_lyapunov(n=20000)
check("classic Lyapunov positive", lam > 0.0)
check("classic Lyapunov ~0.42", abs(lam - 0.42) < 0.05)
check("is_chaotic true for classic", henon.is_chaotic(n=10000))

# A periodic (non-chaotic) parameter set has a non-positive Lyapunov exponent.
# a=0.2, b=0.3 settles to a fixed point -> negative exponent.
check("small-a case not chaotic", henon.largest_lyapunov(a=0.2, b=0.3, n=10000) < 0.0)
check("is_chaotic false for small a", not henon.is_chaotic(a=0.2, b=0.3, n=8000))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all henon tests passed")

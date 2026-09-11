"""Tests for spline.py -- cubic spline interpolation.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Verifies the spline passes
through every knot, is C^2, reproduces cubics, and beats a single polynomial on Runge data.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import spline as S  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- interpolation passes through every knot -------------------------------
xs = [0, 1, 2, 3, 4]
ys = [0, 1, 4, 9, 16]
sp = S.CubicSpline(xs, ys)
check("spline passes through all knots", all(approx(sp(x), y) for x, y in zip(xs, ys)))
check("a two-point spline is the connecting cubic (interpolates ends)",
      approx(S.CubicSpline([0, 2], [1, 5])(0), 1.0) and approx(S.CubicSpline([0, 2], [1, 5])(2), 5.0))
try:
    S.CubicSpline([0, 1], [1])
    check("rejects mismatched lengths", False)
except ValueError:
    check("rejects mismatched lengths", True)
try:
    S.CubicSpline([1], [1])
    check("rejects fewer than two points", False)
except ValueError:
    check("rejects fewer than two points", True)
try:
    S.CubicSpline([0, 0, 1], [1, 2, 3])
    check("rejects non-increasing xs", False)
except ValueError:
    check("rejects non-increasing xs", True)

# --- natural boundary conditions -------------------------------------------
check("natural spline has zero end curvature", approx(sp.M[0], 0.0) and approx(sp.M[-1], 0.0))

# --- C^2 continuity at the interior knots ----------------------------------
eps = 1e-6
cont_ok = True
for k in xs[1:-1]:
    # value, slope, and curvature all match across the join
    if not approx(sp(k - eps), sp(k + eps), 1e-4):
        cont_ok = False
    if not approx(sp.derivative(k - eps), sp.derivative(k + eps), 1e-4):
        cont_ok = False
    if not approx(sp.second_derivative(k - eps), sp.second_derivative(k + eps), 1e-4):
        cont_ok = False
check("spline is C^2 (value, slope, curvature continuous at joins)", cont_ok)

# --- second derivative is piecewise linear ---------------------------------
# at a knot it equals the solved M value
check("second derivative at a knot equals the solved M", approx(sp.second_derivative(2), sp.M[2], 1e-6))

# --- reproduces a cubic exactly (clamped with true end slopes) -------------
f = lambda x: 2 * x ** 3 - x + 1
df = lambda x: 6 * x ** 2 - 1
cx = [0, 1, 2, 3]
cy = [f(x) for x in cx]
spc = S.CubicSpline(cx, cy, bc="clamped", clamp=(df(0), df(3)))
check("clamped spline reproduces a cubic at midpoints",
      all(approx(spc(x), f(x), 1e-8) for x in (0.5, 1.5, 2.5)))
check("clamped spline matches the cubic's derivative",
      all(approx(spc.derivative(x), df(x), 1e-7) for x in (0.5, 1.5, 2.5)))
check("clamped spline honours the end slopes", approx(spc.derivative(0), df(0), 1e-7))

# --- accuracy on a smooth function -----------------------------------------
sx = [i * math.pi / 8 for i in range(9)]
sy = [math.sin(x) for x in sx]
ssp = S.CubicSpline(sx, sy)
check("spline of sin is accurate between knots", approx(ssp(math.pi / 3), math.sin(math.pi / 3), 1e-3))
check("spline derivative approximates cos", approx(ssp.derivative(math.pi / 3), math.cos(math.pi / 3), 1e-2))

# --- beats a single polynomial on Runge's function -------------------------
rx = [-1 + i * 0.2 for i in range(11)]
ry = [1 / (1 + 25 * x * x) for x in rx]
runge = S.CubicSpline(rx, ry)
true_09 = 1 / (1 + 25 * 0.81)
spline_err = abs(runge(0.9) - true_09)
lagrange_err = abs(S.lagrange(rx, ry, 0.9) - true_09)
check("spline stays near the true Runge value", spline_err < 0.02)
check("single polynomial oscillates wildly (Runge)", lagrange_err > 1.0)
check("spline is far more accurate than the polynomial near the edge", spline_err < lagrange_err / 10)

# --- Lagrange still interpolates the knots (sanity) ------------------------
check("Lagrange passes through its knots", all(approx(S.lagrange(rx, ry, x), y) for x, y in zip(rx, ry)))

# --- Thomas solver correctness ---------------------------------------------
# solve a known tridiagonal system: [[2,1,0],[1,2,1],[0,1,2]] x = [3,4,3] -> x=[1,1,1]
sol = S._thomas([0, 1, 1], [2, 2, 2], [1, 1, 0], [3, 4, 3])
check("Thomas algorithm solves a tridiagonal system", all(approx(v, 1.0) for v in sol))

# --- extrapolation uses the end cubics (does not crash) --------------------
check("spline extrapolates below the range", isinstance(sp(-1.0), float))
check("spline extrapolates above the range", isinstance(sp(5.0), float))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall spline tests passed")

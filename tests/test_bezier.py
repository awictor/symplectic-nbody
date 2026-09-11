"""Tests for bezier: de Casteljau, endpoints, convex hull, subdivision, elevation, arc length."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bezier import (de_casteljau, evaluate, bernstein_evaluate, derivative,
                    derivative_control_points, subdivide, elevate_degree, sample,
                    arc_length, in_convex_hull, _lerp)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx_pt(a, b, tol=1e-9):
    return all(abs(a[i] - b[i]) <= tol for i in range(len(a)))


# --- lerp ------------------------------------------------------------------
check("lerp at 0", _lerp((0, 0), (4, 8), 0) == (0, 0))
check("lerp at 1", _lerp((0, 0), (4, 8), 1) == (4, 8))
check("lerp at 0.5", _lerp((0, 0), (4, 8), 0.5) == (2.0, 4.0))

# --- endpoints -------------------------------------------------------------
quad = [(0, 0), (1, 2), (2, 0)]
check("curve starts at the first control point", approx_pt(evaluate(quad, 0), (0, 0)))
check("curve ends at the last control point", approx_pt(evaluate(quad, 1), (2, 0)))
check("quadratic midpoint", approx_pt(evaluate(quad, 0.5), (1.0, 1.0)))

# --- de Casteljau matches the Bernstein sum --------------------------------
cubic = [(0, 0), (1, 3), (3, 3), (4, 0)]
check("de Casteljau == Bernstein (cubic)",
      all(approx_pt(de_casteljau(cubic, t), bernstein_evaluate(cubic, t), 1e-12)
          for t in (0.1, 0.25, 0.5, 0.75, 0.9)))

# --- a linear Bezier is exactly the straight segment -----------------------
lin = [(0, 0), (4, 8)]
check("linear Bezier is the straight line", approx_pt(evaluate(lin, 0.25), (1.0, 2.0)))
check("linear Bezier at 0.5", approx_pt(evaluate(lin, 0.5), (2.0, 4.0)))

# --- derivative ------------------------------------------------------------
dcp = derivative_control_points(lin)
check("derivative of a line is the constant direction", approx_pt(dcp[0], (4, 8)))
check("derivative of a line is constant", approx_pt(derivative(lin, 0.3), (4, 8)))
# quadratic derivative at the apex points horizontally (symmetric arch)
check("symmetric quadratic tangent is horizontal at t=0.5",
      approx_pt(derivative(quad, 0.5), (2.0, 0.0)))

# --- the curve stays inside the convex hull of its control points ----------
check("every curve point is in the control convex hull",
      all(in_convex_hull(evaluate(cubic, i / 40), cubic) for i in range(41)))
# a point far outside is not
check("a far point is outside the hull", not in_convex_hull((100, 100), cubic))

# --- subdivision reproduces the original curve -----------------------------
left, right = subdivide(cubic, 0.5)
check("left half reproduces the first half",
      all(approx_pt(evaluate(left, s), evaluate(cubic, s * 0.5), 1e-12) for s in (0, 0.3, 0.7, 1)))
check("right half reproduces the second half",
      all(approx_pt(evaluate(right, s), evaluate(cubic, 0.5 + s * 0.5), 1e-12)
          for s in (0, 0.3, 0.7, 1)))
check("subdivision preserves the endpoints",
      approx_pt(left[0], cubic[0]) and approx_pt(right[-1], cubic[-1]))
# subdivision at an off-centre t
la, ra = subdivide(cubic, 0.3)
check("off-centre subdivision reproduces the curve",
      all(approx_pt(evaluate(la, s), evaluate(cubic, s * 0.3), 1e-12) for s in (0, 0.5, 1)))

# --- degree elevation preserves the shape ----------------------------------
elevated = elevate_degree(quad)
check("elevation adds one control point", len(elevated) == 4)
check("elevation preserves the curve",
      all(approx_pt(evaluate(elevated, t), evaluate(quad, t), 1e-12)
          for t in (0.1, 0.4, 0.6, 0.9)))
check("elevation keeps the endpoints", approx_pt(elevated[0], quad[0]) and approx_pt(elevated[-1], quad[-1]))
# elevating twice still matches
twice = elevate_degree(elevate_degree(quad))
check("double elevation preserves the curve",
      all(approx_pt(evaluate(twice, t), evaluate(quad, t), 1e-9) for t in (0.2, 0.5, 0.8)))

# --- arc length ------------------------------------------------------------
check("linear arc length equals the endpoint distance", abs(arc_length(lin) - math.hypot(4, 8)) < 1e-6)
# a curve's arc length is at least the chord and at most the control-polygon length
chord = math.dist(cubic[0], cubic[-1])
poly = sum(math.dist(cubic[i], cubic[i + 1]) for i in range(len(cubic) - 1))
L = arc_length(cubic)
check("arc length between chord and control-polygon length", chord <= L <= poly + 1e-6)
# a straight-ish curve (collinear control points) has arc length = span
straight = [(0, 0), (1, 0), (2, 0), (3, 0)]
check("collinear control points -> straight length", abs(arc_length(straight) - 3.0) < 1e-6)

# --- sampling --------------------------------------------------------------
poly_pts = sample(quad, 10)
check("sample returns the requested count", len(poly_pts) == 10)
check("sample endpoints match the curve", approx_pt(poly_pts[0], (0, 0)) and approx_pt(poly_pts[-1], (2, 0)))

# --- 3-D control points work -----------------------------------------------
c3d = [(0, 0, 0), (1, 1, 2), (2, 0, 4)]
check("3-D Bezier endpoints", approx_pt(evaluate(c3d, 0), (0, 0, 0)) and approx_pt(evaluate(c3d, 1), (2, 0, 4)))
check("3-D Bezier midpoint", approx_pt(evaluate(c3d, 0.5), (1.0, 0.5, 2.0)))

# --- a single control point is a constant curve ----------------------------
check("single point is constant", approx_pt(evaluate([(5, 5)], 0.7), (5, 5)))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bezier tests passed")

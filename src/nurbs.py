"""NURBS: Non-Uniform Rational B-Splines, the curves that draw exact circles and conics.

A polynomial B-spline can approximate a circle but never draw one exactly -- no polynomial parameterizes
a circular arc. The fix is to go RATIONAL. Attach a positive WEIGHT w_i to each control point and divide
a weighted B-spline by the sum of the weights:

    C(u) = ( sum_i N_{i,p}(u) w_i P_i ) / ( sum_i N_{i,p}(u) w_i )

The N_{i,p} are the ordinary Cox-de Boor B-spline basis functions, so a NURBS is a B-spline in
homogeneous coordinates projected back down. This one extra degree of freedom per point is exactly what
polynomials lack: with the right weights a quadratic NURBS represents a circle, ellipse, or any conic
EXACTLY. That is why NURBS are the geometry standard for CAD, CAM, and 3D modeling -- one representation
covers freeform surfaces and precise analytic shapes alike.

The rational basis functions R_{i,p}(u) = N_{i,p} w_i / sum_j N_{j,p} w_j inherit the good properties of
the polynomial basis: they are non-negative and form a partition of unity, so the curve stays in the
convex hull of its control points, and a clamped knot vector still makes the curve interpolate its first
and last control points. When all weights are equal the rational terms cancel and a NURBS collapses to
an ordinary B-spline.

This module evaluates NURBS curves (via the projective/homogeneous trick, reusing the repo's B-spline
basis), builds the standard 9-point quadratic circle, and computes the rational basis. It is validated:
the rational basis is a partition of unity and non-negative; with unit weights a NURBS reproduces the
B-spline evaluator to machine precision; the clamped curve interpolates its endpoints; and -- the marquee
test -- the standard NURBS circle lies on the unit circle to machine precision at every parameter, which
no polynomial curve can do. Pure stdlib; the rational companion to the B-spline and Bezier tools."""

from __future__ import annotations

import math

import b_spline


def evaluate(control, weights, degree, u, knots):
    """Evaluate a NURBS curve at parameter u.

    control: list of d-dim control points. weights: positive weight per control point."""
    n_ctrl = len(control)
    dim = len(control[0])
    N = b_spline.all_basis(degree, u, knots, n_ctrl)
    num = [0.0] * dim
    den = 0.0
    for i in range(n_ctrl):
        nw = N[i] * weights[i]
        den += nw
        for c in range(dim):
            num[c] += nw * control[i][c]
    if den == 0:
        return list(control[0])
    return [num[c] / den for c in range(dim)]


def rational_basis(weights, degree, u, knots, n_ctrl):
    """The rational basis functions R_{i,p}(u) = N_i w_i / sum_j N_j w_j."""
    N = b_spline.all_basis(degree, u, knots, n_ctrl)
    den = sum(N[i] * weights[i] for i in range(n_ctrl))
    if den == 0:
        return [0.0] * n_ctrl
    return [N[i] * weights[i] / den for i in range(n_ctrl)]


def sample(control, weights, degree, knots, n_samples=100):
    """Sample the NURBS curve across its valid domain [u_p, u_{n+1}]."""
    n_ctrl = len(control)
    u0 = knots[degree]
    u1 = knots[n_ctrl]
    pts = []
    for k in range(n_samples):
        u = u0 + (u1 - u0) * k / (n_samples - 1)
        pts.append(evaluate(control, weights, degree, u, knots))
    return pts


def circle(center=(0.0, 0.0), radius=1.0):
    """The standard 9-point, degree-2 NURBS representation of a full circle.

    Returns (control_points, weights, degree, knots). The corner control points sit at the square
    circumscribing the circle; the corner weights are cos(45 deg) = sqrt(2)/2, which bends the
    quadratic arcs onto the exact circle."""
    cx, cy = center
    r = radius
    w = math.sqrt(2) / 2.0
    # 9 control points: axis points and corner points around the circle (CCW from +x)
    ctrl = [
        [cx + r, cy],           # +x
        [cx + r, cy + r],       # corner (+x,+y)
        [cx,     cy + r],       # +y
        [cx - r, cy + r],       # corner (-x,+y)
        [cx - r, cy],           # -x
        [cx - r, cy - r],       # corner (-x,-y)
        [cx,     cy - r],       # -y
        [cx + r, cy - r],       # corner (+x,-y)
        [cx + r, cy],           # back to +x
    ]
    weights = [1, w, 1, w, 1, w, 1, w, 1]
    degree = 2
    # knot vector for a 4-arc circle: [0,0,0, 1,1, 2,2, 3,3, 4,4,4] normalized
    knots = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4]
    knots = [k / 4.0 for k in knots]
    return ctrl, weights, degree, knots


def ellipse(center=(0.0, 0.0), a=1.0, b=1.0):
    """A degree-2 NURBS ellipse: the circle construction with independent x/y radii."""
    ctrl, weights, degree, knots = circle(center, 1.0)
    cx, cy = center
    scaled = [[cx + (p[0] - cx) * a, cy + (p[1] - cy) * b] for p in ctrl]
    return scaled, weights, degree, knots

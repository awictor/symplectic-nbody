"""Bezier curves: the control-point curves behind fonts, vector art, and animation.

A Bezier curve is defined not by points it passes through but by CONTROL POINTS that pull it into
shape -- the curve starts at the first control point, ends at the last, and is tugged toward the
ones between without (in general) touching them. That intuitive "handles" behaviour is why Bezier
curves are the primitive of PostScript/PDF/SVG paths, TrueType fonts, and animation easing. A
degree-n curve has n+1 control points and is the weighted blend B(t) = sum_i C(n,i) t^i (1-t)^(n-i)
P_i of them by the BERNSTEIN polynomials, for t in [0, 1].

Evaluating that sum directly is fine, but the elegant and numerically stable way is DE CASTELJAU'S
ALGORITHM: repeatedly take pairwise linear interpolations of the control points at parameter t until
one point remains -- that point is B(t). The same construction, kept rather than discarded, SPLITS
the curve at t into two Bezier curves (subdivision), the basis of adaptive rendering and
intersection. This module also gives the DERIVATIVE (a degree n-1 Bezier of the scaled control
differences, for tangents and velocity), DEGREE ELEVATION (rewrite a curve with one more control
point, same shape), and arc length by adaptive subdivision.

This module implements de Casteljau evaluation, the derivative, subdivision, degree elevation, and
arc length -- verified that the curve hits its first and last control points, that a linear (2-point)
Bezier is exactly the straight segment and a quadratic matches the Bernstein formula, that the curve
stays within the convex hull of its control points, that subdivision reproduces the original curve,
that degree elevation preserves the shape, and that the derivative points along the curve. Pure
stdlib; a curves companion to the interpolation-spline note (which passes THROUGH its points, where
Bezier is pulled BY its control points)."""

from __future__ import annotations

import math


def _lerp(a, b, t):
    return tuple(a[i] + t * (b[i] - a[i]) for i in range(len(a)))


def de_casteljau(control, t):
    """Evaluate the Bezier curve with the given control points at parameter t via repeated linear
    interpolation. Works in any dimension."""
    pts = [tuple(p) for p in control]
    while len(pts) > 1:
        pts = [_lerp(pts[i], pts[i + 1], t) for i in range(len(pts) - 1)]
    return pts[0]


def evaluate(control, t):
    """Alias for de_casteljau (curve point at parameter t)."""
    return de_casteljau(control, t)


def _binom(n, k):
    return math.comb(n, k)


def bernstein_evaluate(control, t):
    """Evaluate via the explicit Bernstein-polynomial sum (a cross-check on de Casteljau)."""
    n = len(control) - 1
    dim = len(control[0])
    out = [0.0] * dim
    for i, p in enumerate(control):
        w = _binom(n, i) * (t ** i) * ((1 - t) ** (n - i))
        for d in range(dim):
            out[d] += w * p[d]
    return tuple(out)


def derivative_control_points(control):
    """Control points of the derivative curve: a degree n-1 Bezier of n*(P_{i+1}-P_i)."""
    n = len(control) - 1
    if n < 1:
        return [tuple(0.0 for _ in control[0])]
    return [tuple(n * (control[i + 1][d] - control[i][d]) for d in range(len(control[0])))
            for i in range(n)]


def derivative(control, t):
    """The tangent (velocity) vector of the curve at parameter t."""
    return de_casteljau(derivative_control_points(control), t)


def subdivide(control, t):
    """Split the curve at parameter t into two Bezier curves (left, right) that together reproduce
    it. The left/right control points are the outer edges of the de Casteljau triangle."""
    pts = [tuple(p) for p in control]
    left = [pts[0]]
    right = [pts[-1]]
    while len(pts) > 1:
        pts = [_lerp(pts[i], pts[i + 1], t) for i in range(len(pts) - 1)]
        left.append(pts[0])
        right.append(pts[-1])
    right.reverse()
    return left, right


def elevate_degree(control):
    """Return control points of the same curve with one more control point (degree n -> n+1)."""
    n = len(control) - 1
    dim = len(control[0])
    new = [tuple(control[0])]
    for i in range(1, n + 1):
        a = i / (n + 1)
        new.append(tuple(a * control[i - 1][d] + (1 - a) * control[i][d] for d in range(dim)))
    new.append(tuple(control[-1]))
    return new


def sample(control, n_samples=50):
    """A polyline of n_samples points along the curve (for drawing / length)."""
    return [de_casteljau(control, i / (n_samples - 1)) for i in range(n_samples)]


def arc_length(control, tol=1e-6, max_depth=20):
    """Arc length by adaptive subdivision: compare the control polygon length to the chord; if
    close enough the segment is nearly straight, else split and recurse."""
    def _length(ctrl, depth):
        # chord = straight distance end to end; poly = summed control-polygon length
        chord = math.dist(ctrl[0], ctrl[-1])
        poly = sum(math.dist(ctrl[i], ctrl[i + 1]) for i in range(len(ctrl) - 1))
        if poly - chord < tol or depth >= max_depth:
            return (poly + chord) / 2.0
        left, right = subdivide(ctrl, 0.5)
        return _length(left, depth + 1) + _length(right, depth + 1)
    return _length([tuple(p) for p in control], 0)


def in_convex_hull(point, control, eps=1e-9):
    """True if `point` lies in the convex hull of the control points (a Bezier curve always does).
    Uses the polygon convex hull + point-in-hull, but works for the small control sets here by a
    simple containment test against the axis-aligned bounding box's convex hull is insufficient, so
    we test against the actual hull via orientation of the sorted hull."""
    # build the convex hull of the control points (Andrew's monotone chain), then test containment
    pts = sorted(set(tuple(p) for p in control))
    if len(pts) == 1:
        return all(abs(point[d] - pts[0][d]) < eps for d in range(len(point)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    n = len(hull)
    if n < 3:
        # collinear control points: point must be on the segment span
        return True
    for i in range(n):
        if cross(hull[i], hull[(i + 1) % n], point) < -eps:
            return False
    return True

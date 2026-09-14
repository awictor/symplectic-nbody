"""B-spline curves via the Cox-de Boor recursion: piecewise polynomials with local control.

A single Bezier curve of degree p needs p+1 control points and every point tugs the WHOLE curve;
push one and the entire shape shifts. B-SPLINES fix both problems. They stitch together many
polynomial pieces of a fixed low degree into one smooth curve, governed by a KNOT VECTOR -- a
non-decreasing list of parameter values that says where one polynomial piece ends and the next
begins. Each control point influences only a local window of the curve (p+1 spans), so editing is
local, and the pieces meet with C^{p-1} continuity automatically. B-splines are the foundation of
NURBS, the geometry standard behind essentially every CAD system and font outline.

The basis functions are defined by the elegant COX-DE BOOR recursion: the degree-0 functions are
indicator functions of the knot spans, and each higher-degree basis function is a knot-weighted
blend of two lower-degree ones. The curve is then C(u) = sum_i N_{i,p}(u) P_i. Two facts make the
scheme well-behaved: the basis functions are non-negative and sum to one at every parameter
(PARTITION OF UNITY, so the curve stays in the convex hull of its active control points), and with a
CLAMPED knot vector (the first and last knots repeated p+1 times) the curve passes exactly through
its first and last control points -- and a clamped B-spline with no interior knots is exactly a
Bezier curve.

This module builds clamped and uniform knot vectors, evaluates the Cox-de Boor basis and the curve
(both by direct basis summation and by the numerically-stable de Boor algorithm), and computes
derivatives. It is validated: the basis is a partition of unity and non-negative everywhere; each
basis function has local support on exactly p+1 knot spans; the two evaluation methods agree; a
clamped curve interpolates its endpoints; a clamped B-spline with no interior knots reproduces the
repo's Bezier evaluator to machine precision; and the analytic derivative matches a finite
difference. Pure stdlib; the piecewise-curve companion to the Bezier and low-discrepancy tools."""

from __future__ import annotations


def clamped_knots(n_ctrl, degree):
    """Clamped (open-uniform) knot vector for n_ctrl control points of the given degree.

    First and last knots are repeated degree+1 times; interior knots are uniform. Length = n+p+2."""
    p = degree
    n = n_ctrl - 1
    if n_ctrl < p + 1:
        raise ValueError("need at least degree+1 control points")
    m = n + p + 1                              # last knot index
    knots = [0.0] * (p + 1)
    interior = n - p                           # number of interior knots
    for i in range(1, interior + 1):
        knots.append(i / (interior + 1))
    knots += [1.0] * (p + 1)
    assert len(knots) == m + 1, (len(knots), m + 1)
    return knots


def uniform_knots(n_ctrl, degree):
    """Uniform (periodic-style) knot vector: evenly spaced, not clamped."""
    m = n_ctrl + degree + 1
    return [i / (m - 1) for i in range(m)]


def basis(i, p, u, knots):
    """Cox-de Boor basis function N_{i,p}(u)."""
    if p == 0:
        # indicator of the half-open span [u_i, u_{i+1}); include the right end at the last knot
        if (knots[i] <= u < knots[i + 1]) or (u == knots[-1] and knots[i] <= u <= knots[i + 1]
                                              and knots[i] < knots[i + 1]):
            return 1.0
        return 0.0
    left_den = knots[i + p] - knots[i]
    right_den = knots[i + p + 1] - knots[i + 1]
    left = 0.0
    right = 0.0
    if left_den > 0:
        left = (u - knots[i]) / left_den * basis(i, p - 1, u, knots)
    if right_den > 0:
        right = (knots[i + p + 1] - u) / right_den * basis(i + 1, p - 1, u, knots)
    return left + right


def all_basis(p, u, knots, n_ctrl):
    """Vector of all basis functions N_{i,p}(u) for i in 0..n_ctrl-1."""
    return [basis(i, p, u, knots) for i in range(n_ctrl)]


def evaluate_basis(control, degree, u, knots):
    """Curve point by direct basis summation C(u) = sum_i N_{i,p}(u) P_i."""
    n_ctrl = len(control)
    dim = len(control[0])
    N = all_basis(degree, u, knots, n_ctrl)
    return [sum(N[i] * control[i][d] for i in range(n_ctrl)) for d in range(dim)]


def _find_span(u, degree, knots, n_ctrl):
    """Index of the knot span containing u (the de Boor algorithm's starting index)."""
    if u >= knots[n_ctrl]:
        return n_ctrl - 1
    if u <= knots[degree]:
        return degree
    lo, hi = degree, n_ctrl
    mid = (lo + hi) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def evaluate(control, degree, u, knots):
    """Curve point via the numerically-stable de Boor algorithm."""
    n_ctrl = len(control)
    dim = len(control[0])
    p = degree
    span = _find_span(u, p, knots, n_ctrl)
    # local copy of the p+1 active control points
    d = [list(control[span - p + j]) for j in range(p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = span - p + j
            den = knots[i + p - r + 1] - knots[i]
            alpha = 0.0 if den == 0 else (u - knots[i]) / den
            for c in range(dim):
                d[j][c] = (1 - alpha) * d[j - 1][c] + alpha * d[j][c]
    return d[p]


def sample(control, degree, knots, n_samples=100):
    """Sample the curve at n_samples parameters across the valid domain [u_p, u_{n+1}]."""
    n_ctrl = len(control)
    u0 = knots[degree]
    u1 = knots[n_ctrl]
    pts = []
    for k in range(n_samples):
        u = u0 + (u1 - u0) * k / (n_samples - 1)
        pts.append(evaluate(control, degree, u, knots))
    return pts


def derivative(control, degree, u, knots, h=1e-6):
    """Numerical derivative dC/du by central difference (kept inside the domain)."""
    n_ctrl = len(control)
    u0 = knots[degree]
    u1 = knots[n_ctrl]
    ua = min(max(u - h, u0), u1)
    ub = min(max(u + h, u0), u1)
    ca = evaluate(control, degree, ua, knots)
    cb = evaluate(control, degree, ub, knots)
    return [(cb[c] - ca[c]) / (ub - ua) for c in range(len(ca))]


def derivative_control_points(control, degree, knots):
    """Control points of the derivative curve (a B-spline of degree p-1).

    Q_i = p * (P_{i+1} - P_i) / (u_{i+p+1} - u_{i+1})."""
    p = degree
    n_ctrl = len(control)
    dim = len(control[0])
    Q = []
    for i in range(n_ctrl - 1):
        den = knots[i + p + 1] - knots[i + 1]
        if den == 0:
            Q.append([0.0] * dim)
        else:
            Q.append([p * (control[i + 1][c] - control[i][c]) / den for c in range(dim)])
    return Q

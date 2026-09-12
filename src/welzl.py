"""Welzl's algorithm: the smallest circle enclosing a set of points, in expected linear time.

Given a cloud of points, the SMALLEST ENCLOSING CIRCLE (also the minimum enclosing disk, or 1-center
problem) is the circle of least radius containing all of them. It is the tightest "where is
everything?" summary: the optimal placement of a facility to minimize the worst-case distance to any
site, the bounding volume for collision culling, and the tolerance circle in metrology. A key fact
makes it tractable: the smallest enclosing circle is determined by at most THREE points on its
boundary (two forming a diameter, or three on the circumcircle), so the answer is pinned down by a
tiny SUPPORT SET.

WELZL'S ALGORITHM (1991) finds it in EXPECTED LINEAR time by a beautiful randomized incremental
method. Process the points in random order, maintaining the smallest circle for those seen so far. If
the next point is already inside the current circle, keep it. If not, that point MUST lie on the
boundary of the new smallest circle -- so recurse, rebuilding the circle from the earlier points with
this point forced onto the boundary. With one, then two, then three boundary points forced, the
circle is determined outright. Because a point is forced to the boundary only rarely (in expectation),
the recursion is shallow and the total work is O(n) expected -- far better than the O(n^4) of checking
every triple.

This module implements Welzl's smallest enclosing circle with a seeded shuffle, returning the centre
and radius. It is verified against exact references and brute force: that every input point lies
inside (or on) the returned circle, that the circle is truly minimal (shrinking the radius excludes a
point, and it matches an O(n^4) all-triples brute-force minimum), that it is defined by 2 or 3
boundary points, and on known cases (two points give a diameter, three give their circumcircle, a
square gives the circumscribed circle). Pure stdlib; a computational-geometry companion to the
convex-hull, rotating-calipers, and Delaunay notes."""

from __future__ import annotations

import math


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _circle_two(a, b):
    """The smallest circle with a and b on its boundary (a diameter): centre = midpoint."""
    cx = (a[0] + b[0]) / 2
    cy = (a[1] + b[1]) / 2
    return (cx, cy), _dist(a, b) / 2


def _circle_three(a, b, c):
    """The circumscribed circle of three points (through all three). None if collinear."""
    ax, ay = a
    bx, by = b
    cx_, cy_ = c
    d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(d) < 1e-14:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy_) + (bx * bx + by * by) * (cy_ - ay)
          + (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx_ - bx) + (bx * bx + by * by) * (ax - cx_)
          + (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d
    centre = (ux, uy)
    return centre, _dist(centre, a)


def _in_circle(p, circle, eps=1e-9):
    if circle is None:
        return False
    centre, r = circle
    return _dist(p, centre) <= r + eps


def _circle_from_boundary(boundary):
    """The circle determined by 0-3 boundary points."""
    if not boundary:
        return None
    if len(boundary) == 1:
        return boundary[0], 0.0
    if len(boundary) == 2:
        return _circle_two(boundary[0], boundary[1])
    # three points
    return _circle_three(*boundary)


class _RNG:
    def __init__(self, seed=1):
        self.state = seed & 0xFFFFFFFF

    def shuffle(self, lst):
        for i in range(len(lst) - 1, 0, -1):
            self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
            j = (self.state >> 8) % (i + 1)
            lst[i], lst[j] = lst[j], lst[i]
        return lst


def smallest_enclosing_circle(points, seed=1):
    """The smallest circle enclosing all points, via Welzl's randomized incremental algorithm.

    Returns (centre, radius). For a single point the radius is 0; for no points, None."""
    pts = [tuple(p) for p in points]
    if not pts:
        return None
    _RNG(seed).shuffle(pts)
    return _welzl_iter(pts)


def _welzl_iter(pts):
    """Iterative move-to-front Welzl (Emo Welzl's practical variant): grow a circle over the points,
    and when a point falls outside, rebuild the circle from the points before it with that point on
    the boundary, moving it to the front. O(n) expected, no deep recursion."""
    circle = None
    for i in range(len(pts)):
        if circle is None or not _in_circle(pts[i], circle):
            circle = _one_boundary(pts, i, pts[i])
    return circle


def _one_boundary(pts, i, p):
    """Smallest circle through the first i points with p forced on the boundary."""
    circle = (p, 0.0)
    for j in range(i):
        if not _in_circle(pts[j], circle):
            circle = _two_boundary(pts, j, p, pts[j])
    return circle


def _two_boundary(pts, j, p, q):
    """Smallest circle through the first j points with p and q forced on the boundary."""
    circle = _circle_two(p, q)
    for k in range(j):
        if not _in_circle(pts[k], circle):
            c3 = _circle_three(p, q, pts[k])
            if c3:
                circle = c3
    return circle


def is_enclosing(points, circle, eps=1e-6):
    """True if `circle` (centre, radius) contains every point."""
    return all(_in_circle(p, circle, eps) for p in points)

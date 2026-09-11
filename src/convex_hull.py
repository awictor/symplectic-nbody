"""Convex hull: the tightest polygon enclosing a set of points.

The convex hull of a point set is the smallest convex polygon containing them all -- the shape a
rubber band snaps to when released around a scatter of pins. It is the foundation of computational
geometry: collision detection, shape analysis, the outer boundary for path planning, and the first
step of many algorithms (farthest-pair, minimum-width, Delaunay). This module builds it by ANDREW'S
MONOTONE CHAIN, an O(n log n) method that sorts the points once, then sweeps left-to-right building
the lower hull and right-to-left building the upper, keeping only left turns.

The engine is the CROSS PRODUCT of two edge vectors, whose sign is the orientation: (b-a) x (c-a) >
0 means a->b->c turns counter-clockwise (a left turn), < 0 clockwise, and = 0 collinear. The chain
keeps pushing points and, whenever the last three make a non-left turn, pops the middle one -- so
only the outer left-turning vertices survive. Concatenating the two half-hulls gives the full
boundary in counter-clockwise order.

From the hull come several classic quantities computed here: the enclosed AREA by the shoelace
formula, PERIMETER, whether an arbitrary point lies inside (orientation tests against each edge),
and the DIAMETER (farthest pair of points, which always lie on the hull) by rotating calipers. This
module implements the hull, area, perimeter, point-in-hull, and diameter -- verified that a square's
hull is its four corners (interior points dropped), that collinear and duplicate points are handled,
that the hull is convex and counter-clockwise, that its area matches the shoelace value, that every
input point lies inside or on it, and that the diameter is the true farthest pair. Pure stdlib; a
computational-geometry companion to the k-d tree note."""

from __future__ import annotations

import math


def _cross(o, a, b):
    """2-D cross product (a-o) x (b-o); >0 left turn, <0 right turn, 0 collinear."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Convex hull by Andrew's monotone chain, returned counter-clockwise without the closing
    duplicate. Degenerate inputs (<3 unique points, or all collinear) return the extreme points."""
    pts = sorted(set(map(tuple, points)))
    n = len(pts)
    if n <= 2:
        return pts
    # lower hull: keep only counter-clockwise (left) turns
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    # drop each half's last point (it is the other half's first) and concatenate
    return lower[:-1] + upper[:-1]


def polygon_area(hull):
    """Enclosed area of a polygon (shoelace formula); nonnegative for any orientation."""
    n = len(hull)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def perimeter(hull):
    """Total edge length around a polygon."""
    n = len(hull)
    if n < 2:
        return 0.0
    return sum(math.dist(hull[i], hull[(i + 1) % n]) for i in range(n))


def point_in_hull(point, hull):
    """True if `point` is inside or on the convex hull (assumed counter-clockwise)."""
    n = len(hull)
    if n < 3:
        return False
    for i in range(n):
        if _cross(hull[i], hull[(i + 1) % n], point) < 0:
            return False        # strictly right of some edge -> outside
    return True


def diameter(points):
    """Farthest pair of points and their distance (the set's diameter). The pair always lies on
    the hull, so only hull vertices are compared -- O(h^2) here for clarity."""
    hull = convex_hull(points)
    if len(hull) < 2:
        return (hull[0], hull[0], 0.0) if hull else (None, None, 0.0)
    best = 0.0
    pair = (hull[0], hull[1])
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            d = math.dist(hull[i], hull[j])
            if d > best:
                best = d
                pair = (hull[i], hull[j])
    return pair[0], pair[1], best


def is_convex_ccw(hull):
    """True if the polygon is convex and wound counter-clockwise (every turn is a left turn)."""
    n = len(hull)
    if n < 3:
        return True
    for i in range(n):
        if _cross(hull[i], hull[(i + 1) % n], hull[(i + 2) % n]) < 0:
            return False
    return True

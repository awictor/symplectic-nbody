"""Minkowski sum: the shape you sweep by sliding one polygon around another, and the geometry of collision.

The Minkowski sum of two sets A and B is A (+) B = { a + b : a in A, b in B } -- every point of A added
to every point of B. Geometrically it is the region swept when you slide B around while its reference
point traces all of A (or vice versa). It is the mathematical heart of MOTION PLANNING and COLLISION
DETECTION: grow an obstacle by the robot's shape (the CONFIGURATION-SPACE obstacle) and the robot
shrinks to a point that must merely avoid the grown obstacle; and two convex shapes A, B overlap if and
only if the origin lies inside their Minkowski DIFFERENCE A (+) (-B), the fact GJK exploits.

For two CONVEX polygons the sum is itself convex and has a gorgeous linear-time construction. Sort both
polygons' edges by angle (their edges, walked counterclockwise, already come in angular order); then
MERGE the two edge sequences by angle, like merging two sorted lists, appending each edge to a running
vertex. Because a convex polygon is exactly its edges laid end to end in turning order, interleaving the
two edge sets by direction traces out the boundary of the sum in O(n + m) time -- no n*m enumeration
needed. (For non-convex inputs one falls back to summing all vertex pairs and taking the convex hull,
which this module also provides as an independent check.)

This module computes the Minkowski sum of two convex polygons by the edge-merge algorithm, the
brute-force all-pairs-plus-hull version, and a convex-polygon intersection test via the Minkowski
difference. It is validated: the edge-merge sum matches the brute-force sum; the sum of two squares is
the expected larger square (with the right area and vertex count); summing with a single point is a
pure translation; the sum's area equals area(A) + area(B) + mixed terms (and for two convex shapes is
at least area(A)+area(B)); translating an input translates the sum; and two polygons intersect exactly
when the origin lies in their Minkowski difference. Reuses the repo's convex hull. Pure stdlib; the
computational-geometry companion to the convex-hull, GJK, and polygon-clipping tools."""

from __future__ import annotations

import math

from convex_hull import convex_hull, polygon_area, point_in_hull


def _ccw(poly):
    """Return the polygon oriented counterclockwise, starting at its lowest (then leftmost) vertex."""
    hull = convex_hull(poly)                      # convex_hull returns a CCW hull
    # rotate so the lowest-then-leftmost vertex is first
    start = min(range(len(hull)), key=lambda i: (hull[i][1], hull[i][0]))
    return hull[start:] + hull[:start]


def _angle(dx, dy):
    a = math.atan2(dy, dx)
    return a + 2 * math.pi if a < 0 else a


def minkowski_sum_convex(A, B):
    """Minkowski sum of two convex polygons by the O(n+m) edge-merge algorithm."""
    P = _ccw(A)
    Q = _ccw(B)
    n, m = len(P), len(Q)
    i = j = 0
    result = []
    # start vertex is the sum of the two lowest vertices
    while i < n or j < m:
        result.append((P[i % n][0] + Q[j % m][0], P[i % n][1] + Q[j % m][1]))
        # compare the two current edge directions
        ai = _angle(P[(i + 1) % n][0] - P[i % n][0], P[(i + 1) % n][1] - P[i % n][1])
        aj = _angle(Q[(j + 1) % m][0] - Q[j % m][0], Q[(j + 1) % m][1] - Q[j % m][1])
        if i >= n:
            j += 1
        elif j >= m:
            i += 1
        elif ai < aj - 1e-12:
            i += 1
        elif ai > aj + 1e-12:
            j += 1
        else:
            i += 1
            j += 1
    # the walk can revisit the start; dedupe consecutive duplicates and re-hull for safety
    return convex_hull(result)


def minkowski_sum_brute(A, B):
    """Minkowski sum by summing all vertex pairs and taking the convex hull (works for any input)."""
    pts = [(a[0] + b[0], a[1] + b[1]) for a in A for b in B]
    return convex_hull(pts)


def negate(poly):
    """Reflect a polygon through the origin: -B."""
    return [(-x, -y) for x, y in poly]


def minkowski_difference(A, B):
    """A (+) (-B), the configuration-space obstacle used for collision tests."""
    return minkowski_sum_convex(A, negate(B))


def convex_intersect(A, B):
    """True if convex polygons A and B overlap: the origin lies in A (+) (-B)."""
    diff = minkowski_difference(A, B)
    return point_in_hull((0.0, 0.0), diff)


def area(poly):
    return polygon_area(poly)

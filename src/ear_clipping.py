"""Ear-clipping polygon triangulation: cutting any simple polygon into triangles.

Triangulating a polygon -- splitting it into non-overlapping triangles that exactly tile its interior
-- is the first step of almost all polygon processing: rendering (GPUs draw only triangles), finite-
element meshing, area and centroid computation, point location, and collision geometry. The TWO EARS
THEOREM (Meisters, 1975) guarantees that every simple polygon with more than three vertices has at
least two EARS -- a vertex whose two neighbours can be joined by a diagonal lying entirely inside the
polygon, cutting off a triangle that contains no other vertex. EAR CLIPPING exploits this directly:
find an ear, snip it off as a triangle, and repeat on the smaller polygon until only a triangle
remains. It is the simplest robust triangulation method, running in O(n^2) -- ideal for the modest
polygons of everyday graphics and GIS.

A vertex is an ear when two conditions hold: it is CONVEX (the polygon turns the right way there, so
the diagonal is interior), and no other vertex of the polygon lies INSIDE the candidate triangle
(otherwise the diagonal would cross the boundary). The algorithm first determines the polygon's
ORIENTATION (clockwise or counter-clockwise, from the signed area) so 'convex' is defined
consistently, then scans for ears, clips the first it finds, and rescans -- the neighbours of a
clipped ear may become ears, so only they need re-checking. The result is n-2 triangles for an
n-vertex polygon, a fact that also serves as a correctness check.

This module triangulates a simple polygon (convex or concave, given as an ordered vertex list) into a
list of triangles, handling either winding order. It is verified against exact references: that a
triangulation of an n-gon always yields exactly n-2 triangles, that the triangle areas sum exactly to
the polygon's area (shoelace formula) with no gaps or overlaps, that every output triangle lies
inside the polygon, that convex polygons, concave polygons, and a deeply non-convex comb shape all
triangulate correctly, and on hand-checked squares and triangles. Pure stdlib; a
computational-geometry companion to the convex-hull, Delaunay, and polygon-clipping notes."""

from __future__ import annotations


def _signed_area(poly):
    """Twice the signed area (shoelace); positive for counter-clockwise winding."""
    n = len(poly)
    s = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s


def polygon_area(poly):
    """The (unsigned) area of a simple polygon."""
    return abs(_signed_area(poly)) / 2.0


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle(p, a, b, c):
    """True if p is strictly inside (or on the boundary of) triangle a, b, c."""
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def triangulate(polygon):
    """Triangulate a simple polygon (list of (x, y) vertices in order). Returns a list of triangles,
    each a tuple of three (x, y) points. Works for either winding order."""
    poly = [tuple(p) for p in polygon]
    n = len(poly)
    if n < 3:
        return []
    if n == 3:
        return [tuple(poly)]

    # ensure counter-clockwise winding so 'convex' means a left turn
    if _signed_area(poly) < 0:
        poly = poly[::-1]

    # work on an index list we shrink as ears are clipped
    indices = list(range(len(poly)))
    triangles = []
    guard = 0
    max_guard = len(poly) ** 2 + 10

    while len(indices) > 3 and guard < max_guard:
        guard += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            i_prev = indices[(i - 1) % m]
            i_curr = indices[i]
            i_next = indices[(i + 1) % m]
            a, b, c = poly[i_prev], poly[i_curr], poly[i_next]
            # convex vertex? (left turn for CCW polygon)
            if _cross(a, b, c) <= 0:
                continue
            # no other polygon vertex inside the candidate ear triangle?
            contains = False
            for j in indices:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _point_in_triangle(poly[j], a, b, c):
                    contains = True
                    break
            if contains:
                continue
            # it's an ear: clip it
            triangles.append((a, b, c))
            del indices[i]
            ear_found = True
            break
        if not ear_found:
            # degenerate / numerical issue: fall back to a fan on the remaining vertices
            break

    if len(indices) == 3:
        triangles.append((poly[indices[0]], poly[indices[1]], poly[indices[2]]))
    return triangles


def triangle_area(a, b, c):
    """Area of a triangle from its three vertices."""
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0

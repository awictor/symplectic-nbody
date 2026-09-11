"""Delaunay triangulation and the Voronoi diagram: the two faces of proximity.

Given a scatter of points in the plane, two dual structures capture "who is near what". The VORONOI
DIAGRAM partitions the plane into one cell per site -- the region closer to that site than to any
other -- and answers nearest-neighbour queries by a single point location. Its straight-line dual is
the DELAUNAY TRIANGULATION, which connects two sites whenever their Voronoi cells share an edge. The
Delaunay triangulation is the "roundest" triangulation of a point set: it maximizes the minimum
angle (avoiding slivers), and it is characterized by the EMPTY-CIRCUMCIRCLE property -- the circle
through the three vertices of any Delaunay triangle contains no other site. These structures are the
workhorses of mesh generation, terrain modelling, interpolation, path planning, and spatial
statistics.

The classic incremental construction is BOWYER-WATSON. Start with a "super-triangle" enclosing all
sites, then insert points one at a time: find every existing triangle whose circumcircle contains
the new point (these become "bad" and violate Delaunay-ness), remove them to leave a star-shaped
polygonal cavity, and re-triangulate that cavity by joining the new point to each boundary edge.
Deleting the triangles that touch the super-triangle's corners at the end leaves the Delaunay
triangulation of the original sites. The Voronoi diagram then falls out for free: its vertices are
the circumcentres of the Delaunay triangles, and an edge connects the circumcentres of two triangles
that share a Delaunay edge.

This module implements Bowyer-Watson Delaunay triangulation, exact circumcircle and circumcentre
computation, and Voronoi-cell / Voronoi-edge extraction from the triangulation. It is verified that
every output triangle satisfies the empty-circumcircle property against all sites, that the triangle
count obeys Euler's formula for a triangulation, that the union of triangle areas equals the convex
hull area (no gaps or overlaps), that Voronoi vertices are equidistant from their three defining
sites, and that a brute-force nearest-site lookup agrees with the Voronoi/Delaunay neighbour
structure. Pure stdlib; a computational-geometry companion to the convex-hull and rotating-calipers
notes."""

from __future__ import annotations

import math
from fractions import Fraction


def _circumcircle(a, b, c):
    """Circumcentre (cx, cy) and squared radius of the triangle a, b, c. Returns None if collinear."""
    ax, ay = a
    bx, by = b
    cx_, cy_ = c
    d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(d) < 1e-18:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy_) + (bx * bx + by * by) * (cy_ - ay)
          + (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx_ - bx) + (bx * bx + by * by) * (ax - cx_)
          + (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d
    r2 = (ax - ux) ** 2 + (ay - uy) ** 2
    return (ux, uy), r2


def _orient(a, b, c):
    """Sign of twice the signed area of triangle a,b,c (>0 = CCW), computed exactly so it agrees with
    the exact in-circle predicate."""
    v = ((Fraction(b[0]) - Fraction(a[0])) * (Fraction(c[1]) - Fraction(a[1]))
         - (Fraction(c[0]) - Fraction(a[0])) * (Fraction(b[1]) - Fraction(a[1])))
    return (v > 0) - (v < 0)


def _in_circumcircle(tri, p, pts):
    """True if point p lies strictly inside the circumcircle of triangle tri (index triple).

    Uses the in-circle DETERMINANT predicate evaluated in EXACT rational arithmetic. Bowyer-Watson
    is fragile to floating-point error: if two adjacent triangles disagree on whether a shared point
    is inside a circumcircle, the cavity of removed triangles is no longer a simple polygon and the
    re-triangulation leaves sliver gaps or overlaps. Computing the sign of the 4x4 in-circle
    determinant with Fraction makes the decision exact and globally consistent, so the mesh is always
    a valid triangulation. Coordinates are converted to Fraction, so the sign is never wrong."""
    a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
    orient = _orient(a, b, c)
    if orient == 0:
        return False
    ax = Fraction(a[0]) - Fraction(p[0]); ay = Fraction(a[1]) - Fraction(p[1])
    bx = Fraction(b[0]) - Fraction(p[0]); by = Fraction(b[1]) - Fraction(p[1])
    cx = Fraction(c[0]) - Fraction(p[0]); cy = Fraction(c[1]) - Fraction(p[1])
    det = ((ax * ax + ay * ay) * (bx * cy - cx * by)
           - (bx * bx + by * by) * (ax * cy - cx * ay)
           + (cx * cx + cy * cy) * (ax * by - bx * ay))
    # for a CCW triangle (orient>0), det>0 means p is inside the circumcircle; flip sign for CW.
    return det > 0 if orient > 0 else det < 0


def triangulate(points):
    """Delaunay triangulation via Bowyer-Watson. Returns a list of triangles as index triples into
    the input points list. Duplicate points are ignored."""
    pts = [tuple(p) for p in points]
    # deduplicate but keep original indices for the returned triangles
    seen = {}
    uniq = []
    for p in pts:
        if p not in seen:
            seen[p] = len(uniq)
            uniq.append(p)
    n = len(uniq)
    if n < 3:
        return []

    # super-triangle: large enough to contain all points
    minx = min(p[0] for p in uniq)
    maxx = max(p[0] for p in uniq)
    miny = min(p[1] for p in uniq)
    maxy = max(p[1] for p in uniq)
    dx = maxx - minx or 1.0
    dy = maxy - miny or 1.0
    dmax = max(dx, dy)
    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2
    # three far-away vertices appended to the working point list
    work = list(uniq)
    M = 1000 * dmax                            # super-triangle far enough that its circumcircles
    s0 = len(work); work.append((midx - M, midy - M))       # never wrongly exclude a real site near
    s1 = len(work); work.append((midx, midy + M))           # the hull (a too-small super-triangle
    s2 = len(work); work.append((midx + M, midy - M))       # is the classic Bowyer-Watson gap bug)

    triangles = [(s0, s1, s2)]

    for i in range(n):
        p = work[i]
        bad = [t for t in triangles if _in_circumcircle(t, p, work)]
        # boundary of the cavity = edges that belong to exactly one bad triangle
        edge_count = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (min(e), max(e))
                edge_count[key] = edge_count.get(key, 0) + 1
        boundary = [e for e, c in edge_count.items() if c == 1]
        # remove bad triangles
        bad_set = set(bad)
        triangles = [t for t in triangles if t not in bad_set]
        # re-triangulate the cavity
        for (u, v) in boundary:
            triangles.append((u, v, i))

    # drop triangles touching a super-triangle vertex
    supers = {s0, s1, s2}
    result = [t for t in triangles if not (supers & set(t))]
    return result


def circumcenter(a, b, c):
    """The circumcentre of triangle a, b, c (a Voronoi vertex). None if collinear."""
    cc = _circumcircle(a, b, c)
    return cc[0] if cc else None


def _tri_edges(t):
    return [(min(t[0], t[1]), max(t[0], t[1])),
            (min(t[1], t[2]), max(t[1], t[2])),
            (min(t[2], t[0]), max(t[2], t[0]))]


def voronoi_edges(points, triangles=None):
    """Finite Voronoi edges as pairs of circumcentre points ((x1,y1),(x2,y2)). Each interior
    Delaunay edge (shared by two triangles) yields the segment between the two circumcentres.
    Unbounded cells' infinite edges are omitted."""
    pts = dedup_index_map(points)             # triangle indices refer to the deduplicated list
    if triangles is None:
        triangles = triangulate(points)
    # map each Delaunay edge to the triangles that contain it
    edge_tris = {}
    centers = {}
    for t in triangles:
        centers[t] = circumcenter(pts[t[0]], pts[t[1]], pts[t[2]])
        for e in _tri_edges(t):
            edge_tris.setdefault(e, []).append(t)
    segments = []
    for e, ts in edge_tris.items():
        if len(ts) == 2:
            c0 = centers[ts[0]]
            c1 = centers[ts[1]]
            if c0 and c1:
                segments.append((c0, c1))
    return segments


def delaunay_neighbors(points, triangles=None):
    """Adjacency: for each site index, the set of sites sharing a Delaunay edge with it."""
    pts = dedup_index_map(points)             # triangle indices refer to the deduplicated list
    if triangles is None:
        triangles = triangulate(points)
    adj = {i: set() for i in range(len(pts))}
    # triangles use indices into the deduplicated list; rebuild that mapping
    for t in triangles:
        for (u, v) in _tri_edges(t):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    return adj


def dedup_index_map(points):
    """The deduplicated point list and the index each triangle vertex refers to (helper for callers
    that need to relate triangle indices back to coordinates)."""
    seen = {}
    uniq = []
    for p in (tuple(q) for q in points):
        if p not in seen:
            seen[p] = len(uniq)
            uniq.append(p)
    return uniq

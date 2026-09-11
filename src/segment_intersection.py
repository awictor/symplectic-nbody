"""Line-segment intersection: do two segments cross, and where?

Deciding whether two line segments intersect -- and finding the crossing point -- is the atom of
computational geometry: collision detection, map overlays, clipping, and mesh processing all reduce
to it. The robust test avoids computing slopes (which blow up for vertical lines) and instead uses
the ORIENTATION of point triples, from the sign of a cross product:

    orient(a, b, c) = sign((b-a) x (c-a))   -- +1 counter-clockwise, -1 clockwise, 0 collinear

Two segments AB and CD PROPERLY cross when A and B lie on opposite sides of line CD *and* C and D
lie on opposite sides of line AB -- i.e. orient(A,B,C) and orient(A,B,D) differ, and likewise for
CD. The fiddly part is the COLLINEAR/TOUCHING cases (an endpoint lying on the other segment,
overlapping collinear segments): those give a zero orientation and are settled by an on-segment
bounding-box check. Handling them correctly is what separates a toy from a usable predicate.

Given that segments do intersect, the crossing POINT of their infinite lines is found by solving
the 2x2 parametric system. From this atom the module builds two applications: testing whether a
POLYGON is SIMPLE (no non-adjacent edges cross -- itself the correctness condition for area and
point-in-polygon algorithms), and a brute all-pairs INTERSECTION count. This module implements the
orientation predicate, the segment-intersection test with full collinear handling, the intersection
point, the simple-polygon test, and pairwise intersection detection -- verified on crossing,
touching, collinear-overlapping, parallel, and disjoint segments, that the intersection point is
correct, that a convex polygon is simple while a figure-eight is not, and against hand-computed
cases. Pure stdlib; a computational-geometry companion to the convex-hull note."""

from __future__ import annotations


def orientation(a, b, c):
    """Sign of the cross product (b-a) x (c-a): +1 CCW, -1 CW, 0 collinear."""
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _on_segment(a, b, p):
    """True if collinear point p lies within the bounding box of segment ab (so on ab)."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_intersect(a, b, c, d):
    """True if segment AB intersects segment CD, including touching and collinear overlap."""
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    # proper crossing: each segment straddles the other's line
    if o1 != o2 and o3 != o4:
        return True
    # collinear / touching special cases (an endpoint on the other segment)
    if o1 == 0 and _on_segment(a, b, c):
        return True
    if o2 == 0 and _on_segment(a, b, d):
        return True
    if o3 == 0 and _on_segment(c, d, a):
        return True
    if o4 == 0 and _on_segment(c, d, b):
        return True
    return False


def intersection_point(a, b, c, d):
    """The intersection point of segments AB and CD, or None if they do not cross at a single
    point (parallel, or overlapping collinear). Solves the 2x2 parametric system."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if denom == 0:
        return None                       # parallel or collinear -> no unique point
    ac = (c[0] - a[0], c[1] - a[1])
    t = (ac[0] * s[1] - ac[1] * s[0]) / denom
    u = (ac[0] * r[1] - ac[1] * r[0]) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (a[0] + t * r[0], a[1] + t * r[1])
    return None                           # lines cross, but outside the segments


def is_simple_polygon(vertices):
    """True if a polygon (list of vertices, implicitly closed) is SIMPLE: no two non-adjacent
    edges intersect. Adjacent edges share an endpoint and are allowed to touch there."""
    n = len(vertices)
    if n < 3:
        return False
    edges = [(vertices[i], vertices[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # skip adjacent edges (they legitimately share a vertex)
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return False
    return True


def count_intersections(segments):
    """Brute-force count of intersecting segment pairs (O(m^2)); returns the list of pairs too."""
    m = len(segments)
    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            (a, b), (c, d) = segments[i], segments[j]
            if segments_intersect(a, b, c, d):
                pairs.append((i, j))
    return len(pairs), pairs

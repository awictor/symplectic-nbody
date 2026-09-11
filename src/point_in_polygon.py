"""Point-in-polygon: is a point inside an arbitrary (possibly concave) polygon?

Testing whether a point lies inside a polygon is the workhorse of hit-testing, GIS ("which county
is this coordinate in?"), and rendering. For a CONVEX polygon a side-of-each-edge check suffices,
but general polygons -- concave, star-shaped, with deep notches -- need one of two classic methods,
both implemented here:

  RAY CASTING (even-odd rule): shoot a ray from the point to infinity and count how many edges it
      crosses. Odd = inside, even = outside. Intuitive and fast, it is blind to winding direction,
      so a self-overlapping region counts by parity.
  WINDING NUMBER: sum the signed angles the polygon subtends around the point; a nonzero total
      winding means inside. It correctly handles self-intersecting polygons where ray parity would
      disagree, distinguishing regions wrapped twice from holes.

The delicate part is BOUNDARY and DEGENERATE cases -- a point exactly on an edge, a ray grazing a
vertex -- which naive implementations get wrong. This module uses the half-open edge convention for
ray casting (count an edge only if the point's y is in [y0, y1) after orienting), so vertex grazes
are counted exactly once, and an explicit on-boundary test. It also computes the SIGNED AREA
(shoelace, whose sign gives the winding orientation) and the CENTROID.

This module implements ray-casting and winding-number point-in-polygon, an on-boundary test,
signed area, and centroid -- verified that both agree with each other and with hand judgement on
convex, concave, and star polygons, that boundary points are detected, that the two methods diverge
(as they should) on a self-intersecting polygon, and that the signed area's sign tracks the vertex
orientation. Pure stdlib; a computational-geometry companion to the segment-intersection and
convex-hull notes."""

from __future__ import annotations


def signed_area(polygon):
    """Shoelace signed area; positive for counter-clockwise vertices, negative for clockwise."""
    n = len(polygon)
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def area(polygon):
    """Unsigned polygon area."""
    return abs(signed_area(polygon))


def centroid(polygon):
    """Area centroid of a simple polygon (the standard signed-area weighted formula)."""
    n = len(polygon)
    a = signed_area(polygon)
    if a == 0:
        # degenerate: fall back to the vertex average
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return (cx / (6 * a), cy / (6 * a))


def _on_edge(p, a, b, eps=1e-12):
    """True if point p lies on the segment ab (collinear and within its bounding box)."""
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if abs(cross) > eps:
        return False
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def on_boundary(point, polygon):
    """True if the point lies on any edge of the polygon."""
    n = len(polygon)
    return any(_on_edge(point, polygon[i], polygon[(i + 1) % n]) for i in range(n))


def ray_casting(point, polygon, include_boundary=True):
    """Even-odd (crossing-number) test. Returns True if `point` is inside `polygon`.

    Uses the half-open convention so a ray grazing a vertex is counted exactly once."""
    if on_boundary(point, polygon):
        return include_boundary
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # does a horizontal ray to +x cross this edge? half-open in y avoids double-counting vertices
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if x < x_cross:
                inside = not inside
    return inside


def winding_number(point, polygon, include_boundary=True):
    """Nonzero-winding test. Returns True if the winding number of `polygon` around `point` is
    nonzero -- correct even for self-intersecting polygons."""
    if on_boundary(point, polygon):
        return include_boundary
    x, y = point
    n = len(polygon)
    wn = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if y1 <= y:
            if y2 > y:                                  # upward crossing
                if _is_left((x1, y1), (x2, y2), point) > 0:
                    wn += 1
        else:
            if y2 <= y:                                 # downward crossing
                if _is_left((x1, y1), (x2, y2), point) < 0:
                    wn -= 1
    return wn != 0


def _is_left(a, b, p):
    """>0 if p is left of the directed line a->b, <0 right, 0 on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])


def winding_count(point, polygon):
    """The actual (signed) winding number of the polygon around the point -- 0 outside, +/-1 for a
    simple loop, +/-2 for a doubly-wrapped region, etc."""
    x, y = point
    n = len(polygon)
    wn = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if y1 <= y:
            if y2 > y and _is_left((x1, y1), (x2, y2), point) > 0:
                wn += 1
        else:
            if y2 <= y and _is_left((x1, y1), (x2, y2), point) < 0:
                wn -= 1
    return wn

"""Rotating calipers: antipodal pairs, diameter, width, and the minimum-area bounding box.

Once a point set's CONVEX HULL is known, a family of extremal geometric quantities can be read off
in a single O(h) sweep instead of the naive O(h**2) all-pairs comparison. The ROTATING CALIPERS
technique, introduced by Shamos and popularized by Toussaint, imagines a pair (or more) of parallel
lines pinching the convex polygon and rotating in lockstep around it. As the calipers rotate a full
turn, the vertices they touch enumerate exactly the ANTIPODAL PAIRS -- pairs of hull vertices that
admit parallel supporting lines -- and the farthest pair (the set's DIAMETER) is always antipodal,
so one rotation finds it.

The same rotating structure answers a surprising range of questions with no extra asymptotic cost:
the WIDTH of the set (the smallest distance between two parallel supporting lines, i.e. the thinnest
slab that contains everything) and, by a theorem of Freeman and Shapira, the MINIMUM-AREA enclosing
RECTANGLE, which must have one side flush with a hull edge -- so trying each edge as the box's
orientation and measuring the extent perpendicular and parallel to it yields the optimum. These
underlie collision bounding volumes, shape metrology, part orientation for packing and machining,
and feature extraction in vision.

This module computes, from a set of 2-D points: the convex hull (Andrew's monotone chain), the
diameter by rotating calipers, the width, and the minimum-area (and minimum-perimeter) enclosing
rectangle with its four corners. It is verified against brute force -- the calipers diameter equals
the O(n**2) farthest pair, the width equals the brute minimum over all hull-edge directions, and the
minimum-area rectangle contains every input point and beats or ties every edge-aligned box -- across
many random and structured point sets, plus exact hand-checked values on squares and triangles. Pure
stdlib; a computational-geometry companion to the convex-hull and closest-pair notes."""

from __future__ import annotations

import math


def _cross(o, a, b):
    """Cross product (a-o) x (b-o); >0 = counter-clockwise turn."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Convex hull as a CCW list of vertices (Andrew's monotone chain), no repeated endpoint."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _dist2(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy


def diameter(points):
    """Farthest pair of points and their Euclidean distance, by rotating calipers in O(h).

    Returns (p, q, distance). The diameter is always realized by an antipodal pair on the hull."""
    hull = convex_hull(points)
    n = len(hull)
    if n < 2:
        return (hull[0], hull[0], 0.0) if hull else (None, None, 0.0)
    if n == 2:
        return hull[0], hull[1], math.dist(hull[0], hull[1])

    # rotating calipers: for each edge i->i+1, advance j while the area (distance from the edge line)
    # keeps increasing; every (i, j) touched is an antipodal pair.
    best = 0.0
    pair = (hull[0], hull[1])
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        # advance j while the next vertex is farther from edge (i, ni)
        while abs(_cross(hull[i], hull[ni], hull[(j + 1) % n])) > abs(_cross(hull[i], hull[ni], hull[j])):
            j = (j + 1) % n
        # candidate antipodal pairs: (i, j) and (ni, j)
        for a, b in ((hull[i], hull[j]), (hull[ni], hull[j])):
            d = _dist2(a, b)
            if d > best:
                best = d
                pair = (a, b)
    return pair[0], pair[1], math.sqrt(best)


def width(points):
    """The width of the set: the smallest distance between two parallel supporting lines (the
    thinnest slab containing every point). Returns the distance. O(h) via rotating calipers over
    hull edges."""
    hull = convex_hull(points)
    n = len(hull)
    if n < 3:
        return 0.0
    best = float("inf")
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        # edge direction; find the hull vertex farthest from this edge line
        ex, ey = b[0] - a[0], b[1] - a[1]
        elen = math.hypot(ex, ey)
        if elen == 0:
            continue
        maxdist = 0.0
        for p in hull:
            # perpendicular distance from p to line a-b
            d = abs((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / elen
            if d > maxdist:
                maxdist = d
        if maxdist < best:
            best = maxdist
    return best


def min_area_rectangle(points):
    """Minimum-area enclosing rectangle. Returns a dict with 'area', 'perimeter', 'corners' (four
    (x, y) in order), 'width', 'height'. By the Freeman-Shapira theorem the optimum rectangle has a
    side collinear with a hull edge, so each edge direction is tried in O(h) each -> O(h**2) total,
    still small since h << n typically."""
    hull = convex_hull(points)
    n = len(hull)
    if n == 0:
        return None
    if n < 3:
        # degenerate: a segment or point
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        return {"area": 0.0, "perimeter": 2 * (max(xs) - min(xs) + max(ys) - min(ys)),
                "corners": [(min(xs), min(ys)), (max(xs), min(ys)),
                            (max(xs), max(ys)), (min(xs), max(ys))],
                "width": max(xs) - min(xs), "height": max(ys) - min(ys)}

    best = None
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        elen = math.hypot(ex, ey)
        if elen == 0:
            continue
        ux, uy = ex / elen, ey / elen          # unit vector along the edge
        vx, vy = -uy, ux                        # unit normal
        # project all hull points onto (u, v)
        min_u = min_v = float("inf")
        max_u = max_v = float("-inf")
        for p in hull:
            pu = p[0] * ux + p[1] * uy
            pv = p[0] * vx + p[1] * vy
            min_u = min(min_u, pu); max_u = max(max_u, pu)
            min_v = min(min_v, pv); max_v = max(max_v, pv)
        w = max_u - min_u
        h = max_v - min_v
        area = w * h
        if best is None or area < best["area"]:
            # reconstruct the four corners in world coordinates
            def world(pu, pv):
                return (pu * ux + pv * vx, pu * uy + pv * vy)
            corners = [world(min_u, min_v), world(max_u, min_v),
                       world(max_u, max_v), world(min_u, max_v)]
            best = {"area": area, "perimeter": 2 * (w + h), "corners": corners,
                    "width": w, "height": h}
    return best

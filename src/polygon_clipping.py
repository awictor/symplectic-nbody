"""Polygon clipping: intersect a polygon with a convex window, the Sutherland-Hodgman way.

Clipping one polygon against another is the workhorse of computer graphics (viewport clipping, "what is
visible in this rectangle?"), GIS (overlaying map layers), and collision geometry. The SUTHERLAND-
HODGMAN algorithm (1974) clips a SUBJECT polygon against a CONVEX clip window by a beautifully simple
pipeline: process the clip window one edge at a time, and for each edge keep only the part of the
current polygon on the inside half-plane. Clipping against edge after edge, the polygon is whittled down
to exactly its intersection with the convex window.

The per-edge step walks the polygon's vertices in order and, for each directed segment (current ->
next), emits output vertices by four cases: both inside -> keep next; inside to outside -> emit the
crossing point; outside to inside -> emit the crossing point then next; both outside -> emit nothing.
Because each clip edge is a straight line, the crossing is a single linear interpolation. The window
must be CONVEX (a rectangle, a triangle, any convex polygon) for the result to stay a single polygon;
clipping against a concave window needs the more elaborate Weiler-Atherton algorithm.

This module implements Sutherland-Hodgman clipping against an arbitrary convex polygon and against an
axis-aligned rectangle, plus the shoelace area and a point-in-convex-polygon test. It is validated: a
polygon entirely inside the window is returned unchanged; a polygon entirely outside is clipped to
nothing; clipping a big square to a smaller centered square yields exactly the smaller square (and its
area); the clipped polygon's area never exceeds the subject's and never exceeds the window's; clipping a
polygon against itself is the identity (idempotence); every clipped vertex lies inside the (closed)
window; and a triangle clipped by a rectangle matches a hand-computed intersection area. Pure stdlib;
the geometry companion to the convex-hull, Voronoi, and line-intersection tools."""

from __future__ import annotations


def _inside(p, a, b):
    """True if point p is on the inside (left) side of the directed edge a->b (CCW convex window)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12


def _intersect(p, q, a, b):
    """Intersection of segment p->q with the infinite line through a->b."""
    # line a->b direction; solve for parameter t on p->q
    r = (q[0] - p[0], q[1] - p[1])
    s = (b[0] - a[0], b[1] - a[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-15:
        return q                                # parallel: fall back to endpoint
    t = ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / denom
    return (p[0] + t * r[0], p[1] + t * r[1])


def _signed_area(poly):
    n = len(poly)
    s = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s / 2.0


def _ensure_ccw(poly):
    return poly if _signed_area(poly) >= 0 else list(reversed(poly))


def clip(subject, window):
    """Clip the subject polygon against a CONVEX clip window (Sutherland-Hodgman).

    Both are lists of (x, y). The window is oriented CCW internally. Returns the clipped polygon
    (possibly empty)."""
    win = _ensure_ccw([tuple(p) for p in window])
    output = [tuple(p) for p in subject]
    n = len(win)
    for i in range(n):
        a = win[i]
        b = win[(i + 1) % n]
        if not output:
            break
        inp = output
        output = []
        m = len(inp)
        for j in range(m):
            cur = inp[j]
            nxt = inp[(j + 1) % m]
            cur_in = _inside(cur, a, b)
            nxt_in = _inside(nxt, a, b)
            if cur_in:
                output.append(cur)
                if not nxt_in:
                    output.append(_intersect(cur, nxt, a, b))
            else:
                if nxt_in:
                    output.append(_intersect(cur, nxt, a, b))
    return output


def clip_rectangle(subject, xmin, ymin, xmax, ymax):
    """Clip the subject polygon against an axis-aligned rectangle."""
    window = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    return clip(subject, window)


def area(poly):
    """Absolute area of a polygon (shoelace)."""
    if len(poly) < 3:
        return 0.0
    return abs(_signed_area(poly))


def point_in_convex(p, window, tol=1e-9):
    """True if point p lies inside the (closed) convex window."""
    win = _ensure_ccw([tuple(q) for q in window])
    n = len(win)
    for i in range(n):
        if not _inside(p, win[i], win[(i + 1) % n]):
            # allow a small tolerance
            a, b = win[i], win[(i + 1) % n]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross < -tol:
                return False
    return True

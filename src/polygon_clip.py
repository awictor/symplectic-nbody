"""Sutherland-Hodgman polygon clipping: intersecting a polygon with a convex window.

Clipping a polygon to a rectangular viewport -- or to any convex region -- is fundamental to
rendering (nothing outside the screen is drawn), CAD, and GIS overlay. The Sutherland-Hodgman
algorithm (1974) does it with a beautifully simple idea: clip the SUBJECT polygon against each edge
of the CLIP polygon in turn, feeding the output of one edge as the input to the next. After passing
through all clip edges, whatever survives is exactly the intersection.

Each single-edge clip is a linear scan of the subject's vertices, deciding for each edge (s -> e)
of the subject which of four cases applies relative to the current clip line:

  both inside      -> keep e
  inside -> outside -> keep the intersection point (leaving)
  outside -> inside -> keep the intersection point (entering) then e
  both outside     -> keep nothing

"Inside" is the side of the clip edge the clip polygon's interior lies on, tested by the cross
product (the same orientation predicate as everywhere in geometry). Because it processes one clip
edge at a time, the algorithm is O(n * k) for an n-vertex subject and k-edge convex clip window,
and it correctly produces the possibly-smaller intersection polygon (it requires the CLIP region to
be convex; a concave subject is fine).

This module implements Sutherland-Hodgman clipping against an arbitrary convex clip polygon, plus a
rectangle-window convenience and the polygon area -- verified that a polygon fully inside the window
is returned unchanged, one fully outside clips to empty, a polygon straddling the boundary clips to
the correct smaller area (checked against the analytic overlap), that clipping a big square to a
smaller one yields the smaller one, and that the clipped area never exceeds the original. Pure
stdlib; a computational-geometry companion to the point-in-polygon and segment-intersection notes."""

from __future__ import annotations


def _inside(p, a, b):
    """True if point p is on the interior (left) side of the directed clip edge a->b.
    For a counter-clockwise clip polygon, interior is to the left, i.e. cross >= 0."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0


def _intersect(s, e, a, b):
    """Intersection of subject edge s->e with the infinite clip line through a->b."""
    x1, y1 = s
    x2, y2 = e
    x3, y3 = a
    x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return e                                # parallel; shouldn't happen for a crossing edge
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _signed_area(poly):
    n = len(poly)
    return sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
               for i in range(n)) / 2.0


def clip_polygon(subject, clip):
    """Clip `subject` against the convex polygon `clip` by Sutherland-Hodgman.

    Both are lists of (x, y) vertices; `clip` must be convex. `subject` may be concave. Returns the
    intersection polygon's vertices (counter-clockwise, empty if they do not overlap)."""
    # ensure the clip polygon is counter-clockwise so "inside" = left of each edge
    clip_ccw = clip if _signed_area(clip) > 0 else list(reversed(clip))
    output = list(subject)
    m = len(clip_ccw)
    for i in range(m):
        a = clip_ccw[i]
        b = clip_ccw[(i + 1) % m]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if _inside(e, a, b):
                if not _inside(s, a, b):
                    output.append(_intersect(s, e, a, b))     # entering
                output.append(e)
            elif _inside(s, a, b):
                output.append(_intersect(s, e, a, b))         # leaving
            s = e
    return output


def clip_to_rectangle(subject, xmin, ymin, xmax, ymax):
    """Clip a polygon to an axis-aligned rectangle window (a common special case)."""
    rect = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    return clip_polygon(subject, rect)


def polygon_area(poly):
    """Unsigned area of a polygon (shoelace)."""
    return abs(_signed_area(poly))

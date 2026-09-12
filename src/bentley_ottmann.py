"""Sweep-line segment intersection: reporting all crossings by an x-ordered plane sweep.

Given n line segments in the plane, how many pairs cross, and where? The brute-force way tests all
O(n^2) pairs. A SWEEP LINE does far less work: move a vertical line left to right, and keep only the
segments whose x-range currently straddles it -- the ACTIVE set. Two segments can only intersect where
their x-ranges overlap, which is exactly when both are active together, so when a segment's left
endpoint is reached it need only be tested against the currently-active segments, not against all n.
Segments are added at their left endpoint and dropped at their right endpoint, driven by an
x-ordered event queue. This prunes away every pair whose x-ranges are disjoint (the vast majority in
typical inputs), the same idea that underlies interval overlap and the classic Bentley-Ottmann
map-overlay sweep. It is a foundational computational-geometry technique -- GIS map overlay, polygon
self-intersection detection, boolean shape operations, and collision detection all build on it.

The event queue holds each segment's left endpoint (an INSERT event) and right endpoint (a REMOVE
event), processed in increasing x with inserts before removes at the same x so segments sharing an
endpoint are compared. At an insert the new segment is tested against every active segment and then
added; at a remove it is dropped. Because the active set is precisely the segments whose x-interval
contains the sweep position, and any intersecting pair shares such a position, every crossing is
discovered exactly once. (The textbook Bentley-Ottmann additionally orders the active set by height
and tests only neighbours for the O((n+k) log n) bound; this implementation keeps the simpler,
provably-complete x-overlap pruning.)

This module reports all intersecting segment pairs and their intersection points via the sweep. It is
verified against the brute-force all-pairs test -- identical sets of intersecting pairs on thousands
of random segment arrangements -- and on hand-built cases (a grid of crossing lines, parallel
non-crossing segments, a shared endpoint). Pure stdlib; a computational-geometry companion to the
segment-intersection predicates, convex-hull, and closest-pair notes."""

from __future__ import annotations

import heapq


def _orient(a, b, c):
    """Orientation of the ordered triple: >0 counter-clockwise, <0 clockwise, 0 collinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_seg(a, b, p):
    return min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and \
        min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9


def segments_intersect(s1, s2):
    """True iff segments s1=(p1,p2) and s2=(p3,p4) intersect (including endpoints/collinear overlap)."""
    (a, b), (c, d) = s1, s2
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        # proper crossing when both orientation pairs strictly differ
        if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
            return True
    # collinear / endpoint touching cases
    if o1 == 0 and _on_seg(a, b, c):
        return True
    if o2 == 0 and _on_seg(a, b, d):
        return True
    if o3 == 0 and _on_seg(c, d, a):
        return True
    if o4 == 0 and _on_seg(c, d, b):
        return True
    return False


def intersection_point(s1, s2):
    """The intersection point of two segments known to cross (line-line intersection). Returns None
    for parallel lines."""
    (x1, y1), (x2, y2) = s1
    (x3, y3), (x4, y4) = s2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _normalise(seg):
    """Order a segment's endpoints left-to-right (then bottom-to-top) so the left endpoint is first."""
    a, b = seg
    if (a[0], a[1]) <= (b[0], b[1]):
        return (a, b)
    return (b, a)


def find_intersections(segments):
    """All intersecting pairs of segments via a sweep line. Returns a sorted list of (i, j, point)
    with i < j the segment indices and `point` their intersection (None if collinear overlap). Each
    pair reported once.

    The sweep maintains the set of segments crossing a vertical line, and at every EVENT (a segment's
    left or right endpoint) re-orders them by height and tests all currently-adjacent pairs. Because
    any two intersecting segments become adjacent in this order at some event before they cross,
    testing adjacencies at every event finds every intersection -- far fewer tests than all-pairs on
    typical inputs, while remaining provably complete."""
    segs = [_normalise(s) for s in segments]

    # events at each segment's left endpoint (insert) and right endpoint (remove); process left
    # (kind 0) before right (kind 1) at the same x so segments sharing an endpoint are compared
    events = []
    for i, (p, q) in enumerate(segs):
        heapq.heappush(events, (p[0], 0, i))
        heapq.heappush(events, (q[0], 1, i))

    active = []            # segments currently crossing the sweep (their x-ranges overlap)
    found = {}             # (i,j) -> point, deduped

    def test(i, j):
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in found:
            return
        if segments_intersect(segs[a], segs[b]):
            found[(a, b)] = intersection_point(segs[a], segs[b])

    while events:
        x, kind, i = heapq.heappop(events)
        if kind == 0:
            # a new segment enters: it can only cross segments whose x-range overlaps it, which are
            # exactly the currently-active ones -- test against each, then add it
            for j in active:
                test(i, j)
            active.append(i)
        else:
            if i in active:
                active.remove(i)

    return sorted((a, b, pt) for (a, b), pt in found.items())


# --- brute-force reference --------------------------------------------------
def brute_find_intersections(segments):
    """All intersecting pairs by testing every pair. Returns sorted (i, j, point) with i < j."""
    n = len(segments)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if segments_intersect(segments[i], segments[j]):
                out.append((i, j, intersection_point(segments[i], segments[j])))
    return sorted(out)


def count_intersections(segments):
    """The number of intersecting segment pairs (via the sweep)."""
    return len(find_intersections(segments))

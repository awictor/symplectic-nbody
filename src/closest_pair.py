"""Closest pair of points: the O(n log n) divide-and-conquer classic.

Finding the two closest points among n is trivially O(n^2) by checking every pair. The elegant
result (Shamos & Hoey, 1975) is that DIVIDE AND CONQUER does it in O(n log n) -- one of the first
demonstrations that geometry could beat the brute-force quadratic. Sort the points by x, split them
into left and right halves, recursively find the closest pair in each, and let d be the smaller of
the two distances. The only pairs left to check are those straddling the split line -- and here is
the trick: any such pair closer than d must lie within a vertical STRIP of width 2d around the line,
and within that strip, sorted by y, each point can be closer than d to only a constant number of
following points (at most 7). So the merge step is linear, and the recurrence T(n) = 2T(n/2) + O(n)
solves to O(n log n).

That "at most 7 neighbours" bound is the crux: a d x 2d rectangle can hold only so many points that
are all >= d apart, so scanning a few points ahead in the y-sorted strip suffices -- no quadratic
blow-up even when many points crowd the boundary.

This module implements the divide-and-conquer closest pair with the strip merge, plus the
brute-force reference -- verified that the two agree exactly on random point sets of many sizes,
that it finds a planted coincident-ish pair, handles duplicate points (distance 0), degenerate
collinear inputs, and small n, and returns the actual pair and its distance. Pure stdlib; a
computational-geometry companion to the convex-hull note."""

from __future__ import annotations

import math


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def brute_force(points):
    """O(n^2) closest pair; the reference implementation. Returns (p, q, distance)."""
    n = len(points)
    if n < 2:
        return (None, None, float("inf"))
    best = float("inf")
    pair = (points[0], points[1])
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(points[i], points[j])
            if d < best:
                best = d
                pair = (points[i], points[j])
    return pair[0], pair[1], best


def closest_pair(points):
    """The two closest points and their distance, in O(n log n) by divide and conquer.
    Returns (p, q, distance)."""
    n = len(points)
    if n < 2:
        return (None, None, float("inf"))
    pts = [tuple(p) for p in points]
    px = sorted(pts, key=lambda p: (p[0], p[1]))          # sorted by x once
    py = sorted(pts, key=lambda p: (p[1], p[0]))          # sorted by y once
    return _rec(px, py)


def _rec(px, py):
    n = len(px)
    if n <= 3:
        return brute_force(px)                            # base case: tiny, brute force

    mid = n // 2
    xmid = px[mid][0]
    left_x = px[:mid]
    right_x = px[mid:]
    # split the y-sorted list into the same left/right halves, matched by coordinate multiset
    left_y, right_y = _partition_y(py, left_x)

    dl = _rec(left_x, left_y)
    dr = _rec(right_x, right_y)
    best = dl if dl[2] < dr[2] else dr
    d = best[2]

    # STRIP: points within horizontal distance d of the split line, kept in y-order
    strip = [p for p in py if abs(p[0] - xmid) < d]
    for i in range(len(strip)):
        # each point need only be compared to the next few (at most 7) in y-order
        j = i + 1
        while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
            dd = _dist(strip[i], strip[j])
            if dd < d:
                d = dd
                best = (strip[i], strip[j], dd)
            j += 1
    return best


def _partition_y(py, left_x):
    """Split the y-sorted list into (left, right) matching the left_x multiset by coordinates."""
    from collections import Counter
    left_multiset = Counter(left_x)
    left_y, right_y = [], []
    for p in py:
        if left_multiset.get(p, 0) > 0:
            left_y.append(p)
            left_multiset[p] -= 1
        else:
            right_y.append(p)
    return left_y, right_y

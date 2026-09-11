"""Quadtrees: recursive spatial partition of the plane for fast region queries.

A quadtree indexes 2-D points by recursively splitting a square region into four equal QUADRANTS
(NW, NE, SW, SE) whenever a node holds more than a small CAPACITY of points. Empty regions stay
shallow and crowded regions subdivide deeply, so the tree adapts to the data's density. It is the
2-D workhorse for spatial queries -- "which points are in this rectangle?", collision broad-phase,
image compression, and the Barnes-Hut n-body approximation -- answering a range query by visiting
only the nodes whose square OVERLAPS the query, pruning whole branches that fall outside.

Where a k-d tree splits alternately on one coordinate at a time (a binary tree), a quadtree splits
on both coordinates at once (a 4-way tree tied to axis-aligned squares), which makes rectangle and
circle range queries and the recursive space-partition especially natural.

This module implements a point-region quadtree with insert, axis-aligned RECTANGLE query, CIRCULAR
range query, nearest-neighbour, and total-count, plus a brute-force reference -- verified that
inserts land in the right quadrant and the tree subdivides past capacity, that rectangle and radius
queries return exactly the same points as a linear scan across many random point sets and queries,
that the nearest-neighbour matches the brute-force nearest, that points outside the root bounds are
rejected, and that the tree depth stays shallow for well-spread data. Pure stdlib; a spatial
data-structure companion to the k-d tree note."""

from __future__ import annotations

import math


class _Rect:
    """An axis-aligned rectangle by centre and half-dimensions (so quadrant splits are exact)."""

    __slots__ = ("cx", "cy", "hw", "hh")

    def __init__(self, cx, cy, hw, hh):
        self.cx, self.cy, self.hw, self.hh = cx, cy, hw, hh

    def contains(self, p):
        return (self.cx - self.hw <= p[0] <= self.cx + self.hw
                and self.cy - self.hh <= p[1] <= self.cy + self.hh)

    def intersects(self, other):
        return not (other.cx - other.hw > self.cx + self.hw
                    or other.cx + other.hw < self.cx - self.hw
                    or other.cy - other.hh > self.cy + self.hh
                    or other.cy + other.hh < self.cy - self.hh)

    def intersects_circle(self, cx, cy, r):
        # closest point of the rect to the circle centre
        dx = max(abs(cx - self.cx) - self.hw, 0)
        dy = max(abs(cy - self.cy) - self.hh, 0)
        return dx * dx + dy * dy <= r * r


class QuadTree:
    """A point-region quadtree over a square/rectangular boundary."""

    def __init__(self, boundary=None, capacity=4, bounds=None, max_depth=24, _depth=0):
        # accept either a _Rect boundary or bounds=(xmin, ymin, xmax, ymax)
        if boundary is None and bounds is not None:
            xmin, ymin, xmax, ymax = bounds
            boundary = _Rect((xmin + xmax) / 2, (ymin + ymax) / 2,
                             (xmax - xmin) / 2, (ymax - ymin) / 2)
        self.boundary = boundary
        self.capacity = capacity
        self.max_depth = max_depth
        self._depth = _depth
        self.points = []          # (x, y) held directly at this node (leaf) or before subdivision
        self.divided = False
        self.nw = self.ne = self.sw = self.se = None

    def _subdivide(self):
        b = self.boundary
        hw, hh = b.hw / 2, b.hh / 2
        d = self._depth + 1
        self.nw = QuadTree(_Rect(b.cx - hw, b.cy + hh, hw, hh), self.capacity, max_depth=self.max_depth, _depth=d)
        self.ne = QuadTree(_Rect(b.cx + hw, b.cy + hh, hw, hh), self.capacity, max_depth=self.max_depth, _depth=d)
        self.sw = QuadTree(_Rect(b.cx - hw, b.cy - hh, hw, hh), self.capacity, max_depth=self.max_depth, _depth=d)
        self.se = QuadTree(_Rect(b.cx + hw, b.cy - hh, hw, hh), self.capacity, max_depth=self.max_depth, _depth=d)
        self.divided = True
        # push existing points down into the children
        existing = self.points
        self.points = []
        for p in existing:
            self._insert_into_children(p)

    def _insert_into_children(self, p):
        for child in (self.nw, self.ne, self.sw, self.se):
            if child.boundary.contains(p):
                return child.insert(p)
        return False

    def insert(self, p):
        """Insert a point; returns True if it lies within the boundary and was stored."""
        if not self.boundary.contains(p):
            return False
        if not self.divided:
            # store here if under capacity, or at max depth (can't split coincident points forever)
            if len(self.points) < self.capacity or self._depth >= self.max_depth:
                self.points.append(p)
                return True
            self._subdivide()
        return self._insert_into_children(p)

    def query_rect(self, cx, cy, hw, hh):
        """All stored points inside the axis-aligned rectangle (centre, half-dims)."""
        rng = _Rect(cx, cy, hw, hh)
        found = []
        self._query_rect(rng, found)
        return found

    def query_rect_bounds(self, xmin, ymin, xmax, ymax):
        """All stored points inside the rectangle given by its corners."""
        return self.query_rect((xmin + xmax) / 2, (ymin + ymax) / 2,
                               (xmax - xmin) / 2, (ymax - ymin) / 2)

    def _query_rect(self, rng, found):
        if not self.boundary.intersects(rng):
            return                                     # prune: no overlap
        for p in self.points:
            if rng.contains(p):
                found.append(p)
        if self.divided:
            for child in (self.nw, self.ne, self.sw, self.se):
                child._query_rect(rng, found)

    def query_circle(self, cx, cy, r):
        """All stored points within distance r of (cx, cy)."""
        found = []
        self._query_circle(cx, cy, r, found)
        return found

    def _query_circle(self, cx, cy, r, found):
        if not self.boundary.intersects_circle(cx, cy, r):
            return
        r2 = r * r
        for p in self.points:
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= r2:
                found.append(p)
        if self.divided:
            for child in (self.nw, self.ne, self.sw, self.se):
                child._query_circle(cx, cy, r, found)

    def nearest(self, cx, cy):
        """The single closest stored point to (cx, cy), or None if the tree is empty."""
        best = [None, float("inf")]
        self._nearest(cx, cy, best)
        return best[0]

    def _nearest(self, cx, cy, best):
        # prune if this node's square cannot beat the current best distance
        if not self.boundary.intersects_circle(cx, cy, math.sqrt(best[1])) and best[0] is not None:
            return
        for p in self.points:
            d2 = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
            if d2 < best[1]:
                best[1] = d2
                best[0] = p
        if self.divided:
            # visit the child containing the query first (better pruning)
            children = sorted((self.nw, self.ne, self.sw, self.se),
                              key=lambda c: (c.boundary.cx - cx) ** 2 + (c.boundary.cy - cy) ** 2)
            for child in children:
                child._nearest(cx, cy, best)

    def count(self):
        """Total number of stored points."""
        n = len(self.points)
        if self.divided:
            n += sum(c.count() for c in (self.nw, self.ne, self.sw, self.se))
        return n

    def depth(self):
        """Maximum depth of the tree (a leaf-only tree is depth 0)."""
        if not self.divided:
            return 0
        return 1 + max(c.depth() for c in (self.nw, self.ne, self.sw, self.se))

    def all_points(self):
        out = list(self.points)
        if self.divided:
            for c in (self.nw, self.ne, self.sw, self.se):
                out.extend(c.all_points())
        return out


def brute_rect(points, xmin, ymin, xmax, ymax):
    """Reference: points inside a rectangle by linear scan."""
    return [p for p in points if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax]


def brute_circle(points, cx, cy, r):
    """Reference: points within radius r by linear scan."""
    return [p for p in points if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= r * r]


def brute_nearest(points, cx, cy):
    """Reference: nearest point by linear scan."""
    if not points:
        return None
    return min(points, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

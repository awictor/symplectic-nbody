"""k-d tree: fast nearest-neighbour search in space.

Given a cloud of points, "which is closest to this query?" is asked constantly -- in graphics,
robotics, machine learning (k-NN), particle simulation, and geographic search. Checking every
point is O(n) per query; a k-d tree (Bentley, 1975) organizes the points so the typical query
costs O(log n).

The idea is a binary space partition. At the root, split the points by their median along the
first coordinate (x); the left subtree holds points with smaller x, the right subtree larger.
At the next level split by y, then z, then cycle back -- each level cuts space by a different
axis. A nearest-neighbour query descends to the leaf containing the query, then unwinds: at
each node it checks whether the splitting plane is closer than the best distance found so far,
and only recurses into the far subtree if that hyper-rectangle could contain something nearer.
Whole branches are pruned, so most of the tree is never visited.

The same descent-and-prune gives k-nearest-neighbours (keep the k best) and range queries (all
points within a radius). It degrades to O(n) in high dimensions -- the curse of dimensionality --
but for the 2D/3D/low-d spaces it was built for it is the standard tool.

This module builds a balanced k-d tree by recursive median splitting, and does exact nearest,
k-nearest, and radius queries, each checked against a brute-force scan. Pure stdlib; the
spatial-data-structure companion to the Dijkstra and Fenwick notes.
"""

from __future__ import annotations

import math


class _Node:
    __slots__ = ("point", "axis", "left", "right")

    def __init__(self, point, axis, left, right):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


def _dist2(a, b):
    """Squared Euclidean distance (avoids sqrt for comparisons)."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


class KDTree:
    """A k-d tree over a list of points (tuples of equal dimension)."""

    def __init__(self, points):
        pts = [tuple(p) for p in points]
        if pts:
            self.k = len(pts[0])
            if any(len(p) != self.k for p in pts):
                raise ValueError("all points must have the same dimension")
        else:
            self.k = 0
        self.size = len(pts)
        self.root = self._build(pts, depth=0)

    def _build(self, pts, depth):
        if not pts:
            return None
        axis = depth % self.k
        pts.sort(key=lambda p: p[axis])
        mid = len(pts) // 2
        return _Node(
            point=pts[mid],
            axis=axis,
            left=self._build(pts[:mid], depth + 1),
            right=self._build(pts[mid + 1:], depth + 1),
        )

    def nearest(self, query):
        """Return the point in the tree closest to `query` (Euclidean). None if the tree is
        empty."""
        if self.root is None:
            return None
        best = [None, math.inf]  # [point, squared distance]
        self._nearest(self.root, tuple(query), best)
        return best[0]

    def _nearest(self, node, query, best):
        if node is None:
            return
        d2 = _dist2(node.point, query)
        if d2 < best[1]:
            best[1] = d2
            best[0] = node.point
        axis = node.axis
        diff = query[axis] - node.point[axis]
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._nearest(near, query, best)
        # only cross the splitting plane if it could hold something closer
        if diff * diff < best[1]:
            self._nearest(far, query, best)

    def k_nearest(self, query, k: int):
        """Return the k points closest to `query`, sorted nearest-first."""
        if k <= 0 or self.root is None:
            return []
        # keep a list of (squared distance, point); simple since k is usually small
        best = []  # list of [d2, point], kept sorted, length <= k

        def consider(point):
            d2 = _dist2(point, query)
            if len(best) < k:
                best.append([d2, point])
                best.sort(key=lambda e: e[0])
            elif d2 < best[-1][0]:
                best[-1] = [d2, point]
                best.sort(key=lambda e: e[0])

        def recurse(node):
            if node is None:
                return
            consider(node.point)
            axis = node.axis
            diff = query[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            recurse(near)
            if len(best) < k or diff * diff < best[-1][0]:
                recurse(far)

        query = tuple(query)
        recurse(self.root)
        return [p for _, p in best]

    def within_radius(self, query, radius: float):
        """All points within `radius` of `query` (inclusive), unordered."""
        if self.root is None:
            return []
        r2 = radius * radius
        query = tuple(query)
        found = []

        def recurse(node):
            if node is None:
                return
            if _dist2(node.point, query) <= r2:
                found.append(node.point)
            axis = node.axis
            diff = query[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            recurse(near)
            if diff * diff <= r2:
                recurse(far)

        recurse(self.root)
        return found

    def height(self):
        """Height of the tree (a balanced build gives ~log2 n)."""
        def h(node):
            return 0 if node is None else 1 + max(h(node.left), h(node.right))
        return h(self.root)


# --- brute-force references (for validation) --------------------------------

def brute_nearest(points, query):
    """Linear-scan nearest neighbour, for checking the tree."""
    best, bd = None, math.inf
    for p in points:
        d = _dist2(p, query)
        if d < bd:
            bd, best = d, tuple(p)
    return best


def brute_k_nearest(points, query, k):
    """Linear-scan k nearest, nearest-first."""
    ordered = sorted((tuple(p) for p in points), key=lambda p: _dist2(p, query))
    return ordered[:k]


def brute_within_radius(points, query, radius):
    """Linear-scan radius query."""
    r2 = radius * radius
    return [tuple(p) for p in points if _dist2(p, query) <= r2]

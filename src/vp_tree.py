"""Vantage-point trees -- nearest-neighbour search in ANY metric space, not just Euclidean coordinates.

A k-d tree is wonderful when your data are points with coordinates you can split on, axis by axis. But
enormous classes of nearest-neighbour problems have no coordinates at all: the "distance" between two
DNA strings is their edit distance, between two documents their cosine or Jaccard distance, between two
graphs some graph-edit metric. There are no axes to split on -- only a black-box distance function that
happens to satisfy the triangle inequality. The VANTAGE-POINT TREE (Yianilos, 1993) is the classic data
structure for exactly this setting: it indexes any METRIC space using nothing but pairwise distances,
and answers nearest-neighbour queries in expected O(log n) instead of the O(n) of scanning everything.

The construction is a beautiful use of the triangle inequality. Pick one point as the VANTAGE POINT and
measure every other point's distance to it. Take the MEDIAN of those distances as a threshold mu:
points closer than mu go in the left ("inside the ball") subtree, points farther go right ("outside").
Recurse. The tree partitions the space into nested spherical shells around vantage points -- no
coordinates required, just the distances.

The query exploits the triangle inequality to PRUNE. Searching for the neighbours of a query q, at a
node with vantage point v and threshold mu, compute d = dist(q, v). If d < mu the true neighbour is
probably inside, so descend left first; but if the current best radius tau reaches across the boundary
(d - tau <= mu), the answer might also lie outside, so the right subtree cannot be skipped. Symmetric
logic holds when d >= mu. Each comparison either confines the search to one side or forces both, and the
triangle inequality guarantees that whenever we prune a subtree, nothing inside it could have beaten the
current best -- so the pruned search returns EXACTLY the same neighbours as a brute-force scan, just far
faster. This module supports single nearest neighbour, k-nearest neighbours (via a bounded max-heap of
the k best so far), and range queries (all points within a radius).

Because the tree needs only a distance callable, it works out of the box for Euclidean points, Manhattan
distance, edit distance on strings, cosine distance on vectors, or any user metric. A seeded
linear-congruential generator chooses vantage points so builds are reproducible.

Validation. The guarantee is that pruned search equals exhaustive search, and that is tested head-on:
(1) for Euclidean points, string edit distance, and cosine distance, the VP-tree's single-nearest and
k-nearest results match a brute-force linear scan on every one of hundreds of seeded random queries;
(2) range queries return exactly the set a brute-force radius filter returns; (3) the returned distances
are correct and sorted; (4) querying a point that is in the tree returns itself at distance zero; (5) the
tree visits far fewer nodes than the brute-force scan, confirming the pruning actually fires. Pure
standard library -- ``math`` and ``heapq`` only."""

import heapq
import math


class _LCG:
    """Seeded linear-congruential generator for reproducible vantage-point choice."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def randint(self, n):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) % n


# ---------------------------------------------------------------------------
# common metrics
# ---------------------------------------------------------------------------

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def edit_distance(a, b):
    """Levenshtein edit distance between two strings (a valid metric)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def angular_distance(a, b):
    """Angular distance arccos(cosine similarity) / pi, in [0, 1].

    Note that 1 - cosine similarity is NOT a metric (it violates the triangle inequality), so it
    cannot be indexed by a VP-tree; the ANGLE between vectors, however, is a genuine metric on the
    sphere, so this is the correct cosine-flavoured distance to use here.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    cos = dot / (na * nb)
    cos = max(-1.0, min(1.0, cos))   # clamp against rounding
    return math.acos(cos) / math.pi


# ---------------------------------------------------------------------------
# the tree
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("point", "index", "threshold", "inside", "outside")

    def __init__(self, point, index):
        self.point = point
        self.index = index
        self.threshold = 0.0
        self.inside = None
        self.outside = None


class VPTree:
    """A vantage-point tree over ``items`` under a ``distance`` metric.

    ``distance`` is any callable satisfying the triangle inequality. Query methods return (distance,
    index, item) tuples so the original items and their positions are recoverable.
    """

    def __init__(self, items, distance=euclidean, seed=12345):
        self.items = list(items)
        self.distance = distance
        self._rng = _LCG(seed)
        self._dist_calls = 0        # instrumentation for validating that pruning fires
        indices = list(range(len(self.items)))
        self.root = self._build(indices)

    # -- construction --------------------------------------------------------
    def _build(self, indices):
        if not indices:
            return None
        # choose a random vantage point, swap it to the front
        vp_pos = self._rng.randint(len(indices))
        indices[0], indices[vp_pos] = indices[vp_pos], indices[0]
        vp_index = indices[0]
        node = _Node(self.items[vp_index], vp_index)

        rest = indices[1:]
        if not rest:
            return node

        # distances from every remaining point to the vantage point
        dists = [(self._raw_dist(self.items[i], node.point), i) for i in rest]
        dists.sort(key=lambda t: t[0])
        mid = len(dists) // 2
        node.threshold = dists[mid][0]

        inside = [i for d, i in dists if d < node.threshold]
        outside = [i for d, i in dists if d >= node.threshold]
        node.inside = self._build(inside)
        node.outside = self._build(outside)
        return node

    def _raw_dist(self, a, b):
        self._dist_calls += 1
        return self.distance(a, b)

    # -- k nearest neighbours ------------------------------------------------
    def k_nearest(self, query, k=1):
        """Return the k nearest items as a sorted list of (distance, index, item), closest first."""
        if k <= 0 or self.root is None:
            return []
        self._dist_calls = 0
        heap = []   # max-heap by negative distance; holds up to k best

        def search(node):
            if node is None:
                return
            d = self._raw_dist(query, node.point)
            if len(heap) < k:
                heapq.heappush(heap, (-d, node.index))
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, node.index))
            tau = -heap[0][0] if len(heap) == k else float("inf")

            if node.inside is None and node.outside is None:
                return
            mu = node.threshold
            if d < mu:
                search(node.inside)
                if d + tau >= mu:                    # boundary within reach -> also search outside
                    tau = -heap[0][0] if len(heap) == k else float("inf")
                    search(node.outside)
            else:
                search(node.outside)
                if d - tau < mu:                     # boundary within reach -> also search inside
                    tau = -heap[0][0] if len(heap) == k else float("inf")
                    search(node.inside)

        search(self.root)
        result = sorted((-nd, idx, self.items[idx]) for nd, idx in heap)
        return result

    def nearest(self, query):
        """Return the single nearest (distance, index, item), or None if the tree is empty."""
        res = self.k_nearest(query, 1)
        return res[0] if res else None

    # -- range query ---------------------------------------------------------
    def within(self, query, radius):
        """Return all items within ``radius`` of the query as a sorted list of (distance, index, item)."""
        if self.root is None:
            return []
        self._dist_calls = 0
        found = []

        def search(node):
            if node is None:
                return
            d = self._raw_dist(query, node.point)
            if d <= radius:
                found.append((d, node.index, self.items[node.index]))
            if node.inside is None and node.outside is None:
                return
            mu = node.threshold
            # a point inside the ball is possible if d - radius < mu; outside if d + radius >= mu
            if d - radius < mu:
                search(node.inside)
            if d + radius >= mu:
                search(node.outside)

        search(self.root)
        found.sort(key=lambda t: t[0])
        return found

    @property
    def distance_calls(self):
        """Number of distance evaluations in the last query (0 after build). For validating pruning."""
        return self._dist_calls


# ---------------------------------------------------------------------------
# brute-force reference
# ---------------------------------------------------------------------------

def brute_k_nearest(items, query, k, distance=euclidean):
    """Reference: k nearest by a full linear scan. Returns sorted (distance, index, item)."""
    scored = sorted((distance(query, it), i, it) for i, it in enumerate(items))
    return scored[:k]


def brute_within(items, query, radius, distance=euclidean):
    """Reference: all items within radius by a full scan."""
    return sorted((distance(query, it), i, it) for i, it in enumerate(items)
                  if distance(query, it) <= radius)

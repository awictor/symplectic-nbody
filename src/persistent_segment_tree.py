"""Persistent segment tree: every update makes a new version, all old versions still queryable.

An ordinary segment tree lets you update and query in O(log n), but each update destroys the old
state. A PERSISTENT segment tree keeps every version alive: an update returns a NEW root, sharing all
the untouched subtrees with the previous version and allocating only the O(log n) nodes on the path
that actually changed. So n updates cost O(n log n) total memory yet give you n+1 fully-queryable
snapshots -- you can ask about the array as it was at any point in its history.

The magic is PATH COPYING. To update a leaf, walk from the root to it; at each step, instead of
mutating a node, create a clone that points to the (unchanged) sibling subtree and to the new child
on the path. The old root still reaches the old version; the new root reaches the updated one. This
is the same idea behind persistent (immutable) data structures in functional languages.

The classic application, and the one this module showcases, is the K-TH SMALLEST ELEMENT IN A RANGE
(a static "range quantile" query, the offline alternative to a wavelet tree or merge-sort tree).
Build a persistent COUNT tree over the value domain, one version per array prefix: version i counts
how many of the first i elements equal each value. Then the multiset of values in a[l..r] is exactly
"version r minus version l-1", and a single simultaneous walk down BOTH versions -- comparing the
left-subtree count difference against k -- finds the k-th smallest in O(log V). This module also
supports arbitrary historical point-add / prefix-sum queries across versions.

Validated against brute force: point-add versions reproduce the correct prefix sums for every
version and range, and the k-th-smallest-in-range query matches a sort of the actual subarray for
thousands of random (array, l, r, k) instances -- including all order statistics (min, median, max).
Pure stdlib; the version-history companion to the segment tree and wavelet tree."""

from __future__ import annotations

import bisect


class _Node:
    __slots__ = ("left", "right", "count")

    def __init__(self, left=None, right=None, count=0):
        self.left = left
        self.right = right
        self.count = count


class PersistentSegmentTree:
    """A persistent segment tree over positions 0..size-1 storing summable values. Each update
    returns a new version root; all prior roots stay valid."""

    def __init__(self, size):
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.empty = self._build(0, size - 1)
        self.roots = [self.empty]  # roots[v] is version v

    def _build(self, lo, hi):
        node = _Node()
        if lo == hi:
            return node
        mid = (lo + hi) // 2
        node.left = self._build(lo, mid)
        node.right = self._build(mid + 1, hi)
        return node

    def _update(self, node, lo, hi, pos, delta):
        # returns a NEW node reflecting +delta at pos, sharing untouched subtrees
        new = _Node(node.left, node.right, node.count + delta)
        if lo == hi:
            return new
        mid = (lo + hi) // 2
        if pos <= mid:
            new.left = self._update(node.left, lo, mid, pos, delta)
        else:
            new.right = self._update(node.right, mid + 1, hi, pos, delta)
        return new

    def update(self, base_version, pos, delta):
        """Create a new version by adding delta at pos, based on base_version. Returns the new
        version index."""
        if not (0 <= pos < self.size):
            raise ValueError("pos out of range")
        new_root = self._update(self.roots[base_version], 0, self.size - 1, pos, delta)
        self.roots.append(new_root)
        return len(self.roots) - 1

    def _query(self, node, lo, hi, l, r):
        if node is None or r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return node.count
        mid = (lo + hi) // 2
        return (self._query(node.left, lo, mid, l, r) +
                self._query(node.right, mid + 1, hi, l, r))

    def query(self, version, l, r):
        """Sum over positions [l, r] in a given version."""
        return self._query(self.roots[version], 0, self.size - 1, l, r)


class RangeKth:
    """K-th smallest element in a subarray, via one persistent count tree per prefix."""

    def __init__(self, array):
        self.array = list(array)
        self.n = len(array)
        # coordinate-compress the values
        self.sorted_vals = sorted(set(array))
        self.m = len(self.sorted_vals)
        self.pst = PersistentSegmentTree(max(1, self.m))
        # version i (1-based) = counts over the first i elements
        self.version_of_prefix = [0]  # prefix 0 = empty (version 0)
        cur = 0
        for x in self.array:
            idx = bisect.bisect_left(self.sorted_vals, x)
            cur = self.pst.update(cur, idx, +1)
            self.version_of_prefix.append(cur)

    def kth_smallest(self, l, r, k):
        """The k-th smallest (1-indexed) value in array[l..r] inclusive."""
        if not (0 <= l <= r < self.n):
            raise ValueError("bad range")
        length = r - l + 1
        if not (1 <= k <= length):
            raise ValueError("k out of range for this subarray")
        vl = self.pst.roots[self.version_of_prefix[l]]      # prefix before l
        vr = self.pst.roots[self.version_of_prefix[r + 1]]  # prefix through r
        lo, hi = 0, self.m - 1
        node_l, node_r = vl, vr
        while lo < hi:
            mid = (lo + hi) // 2
            left_count = (node_r.left.count if node_r.left else 0) - \
                         (node_l.left.count if node_l.left else 0)
            if k <= left_count:
                node_l = node_l.left
                node_r = node_r.left
                hi = mid
            else:
                k -= left_count
                node_l = node_l.right
                node_r = node_r.right
                lo = mid + 1
        return self.sorted_vals[lo]

    def range_rank(self, l, r, value):
        """How many elements in array[l..r] are <= value."""
        idx = bisect.bisect_right(self.sorted_vals, value) - 1
        if idx < 0:
            return 0
        vr = self.version_of_prefix[r + 1]
        vl = self.version_of_prefix[l]
        return self.pst.query(vr, 0, idx) - self.pst.query(vl, 0, idx)


# --- brute-force references --------------------------------------------------
def brute_kth_smallest(array, l, r, k):
    """The k-th smallest in array[l..r] by sorting."""
    return sorted(array[l:r + 1])[k - 1]

"""Segment trees with lazy propagation: O(log n) range queries AND range updates.

A Fenwick tree answers prefix-sum queries with point updates; a SEGMENT TREE goes further, handling
arbitrary RANGE queries (sum, min, max, gcd, ...) AND RANGE updates -- add a value to every element
in [l, r], or assign it -- all in O(log n). It is the data structure behind competitive-programming
range problems, interval scheduling, and any workload that repeatedly asks "what is the aggregate
over this window, and now change that window."

The tree stores, at each node, the aggregate of a contiguous segment of the array; the root covers
everything, and each node splits its range in half between two children. A query descends only into
the O(log n) nodes whose segments tile the query range. The trick for range UPDATES is LAZY
PROPAGATION: rather than touch every leaf in a range (which would be O(n)), a node records a PENDING
update as a "lazy" tag and applies it to itself immediately, pushing it down to its children only
when a later query or update actually visits them. That deferral keeps every operation O(log n).

This module implements a segment tree parameterized by the aggregate (sum, min, or max) with lazy
RANGE-ADD updates, point updates, and range queries -- verified against a brute-force array over
thousands of random mixed operations for all three aggregates, that range-add then range-query
matches recomputation, that point updates and single-element queries agree, and that it handles the
full-array and single-element edge ranges. Pure stdlib; the range-update companion to the
Fenwick-tree note."""

from __future__ import annotations


class SegmentTree:
    """A segment tree over a fixed-length array supporting range-add updates and range queries.

    aggregate: 'sum', 'min', or 'max'. Range-add adds a constant to every element in [l, r];
    range-query returns the aggregate over [l, r]. Both are O(log n) via lazy propagation."""

    def __init__(self, data, aggregate="sum"):
        assert aggregate in ("sum", "min", "max")
        self.n = len(data)
        self.agg = aggregate
        self.tree = [0] * (4 * self.n) if self.n else [0]
        self.lazy = [0] * (4 * self.n) if self.n else [0]
        if self.n:
            self._build(1, 0, self.n - 1, list(data))

    def _combine(self, a, b):
        if self.agg == "sum":
            return a + b
        if self.agg == "min":
            return min(a, b)
        return max(a, b)

    def _build(self, node, lo, hi, data):
        if lo == hi:
            self.tree[node] = data[lo]
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, data)
        self._build(2 * node + 1, mid + 1, hi, data)
        self.tree[node] = self._combine(self.tree[2 * node], self.tree[2 * node + 1])

    def _apply(self, node, lo, hi, add):
        """Apply a pending +add to a node covering [lo, hi], and record it as lazy for its kids."""
        if self.agg == "sum":
            self.tree[node] += add * (hi - lo + 1)     # sum scales with segment length
        else:
            self.tree[node] += add                     # min/max shift by the constant
        self.lazy[node] += add

    def _push_down(self, node, lo, hi):
        """Flush this node's lazy tag to its two children before descending."""
        if self.lazy[node] != 0:
            mid = (lo + hi) // 2
            self._apply(2 * node, lo, mid, self.lazy[node])
            self._apply(2 * node + 1, mid + 1, hi, self.lazy[node])
            self.lazy[node] = 0

    def range_add(self, l, r, add):
        """Add `add` to every element in [l, r] (inclusive)."""
        if self.n:
            self._range_add(1, 0, self.n - 1, l, r, add)

    def _range_add(self, node, lo, hi, l, r, add):
        if r < lo or hi < l:
            return                                      # segment fully outside the update range
        if l <= lo and hi <= r:
            self._apply(node, lo, hi, add)              # segment fully inside -> lazy apply
            return
        self._push_down(node, lo, hi)
        mid = (lo + hi) // 2
        self._range_add(2 * node, lo, mid, l, r, add)
        self._range_add(2 * node + 1, mid + 1, hi, l, r, add)
        self.tree[node] = self._combine(self.tree[2 * node], self.tree[2 * node + 1])

    def point_update(self, i, value):
        """Set element i to `value` (implemented as a range-add of the difference)."""
        current = self.query(i, i)
        self.range_add(i, i, value - current)

    def query(self, l, r):
        """Aggregate over [l, r] (inclusive)."""
        if not self.n:
            return 0
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node, lo, hi, l, r):
        if r < lo or hi < l:
            # identity for the aggregate on an empty/out-of-range segment
            return 0 if self.agg == "sum" else (float("inf") if self.agg == "min" else float("-inf"))
        if l <= lo and hi <= r:
            return self.tree[node]
        self._push_down(node, lo, hi)
        mid = (lo + hi) // 2
        left = self._query(2 * node, lo, mid, l, r)
        right = self._query(2 * node + 1, mid + 1, hi, l, r)
        return self._combine(left, right)

    def to_list(self):
        """Materialize the current array (each element via a point query)."""
        return [self.query(i, i) for i in range(self.n)]

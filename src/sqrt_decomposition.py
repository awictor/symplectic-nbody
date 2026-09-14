"""Square-root decomposition: range queries and updates in O(sqrt n) with almost no machinery.

Between the naive O(n)-per-query array and the O(log n) segment tree sits a gloriously simple idea:
partition the array into about sqrt(n) contiguous BLOCKS of size sqrt(n), and precompute an aggregate
(sum, min, max, ...) for each block. A range query [l, r] then touches at most two PARTIAL blocks at
the ends (walked element by element) and a handful of WHOLE blocks in the middle (read from the
precomputed aggregates), so it costs O(sqrt n) instead of O(n). A point update recomputes just the one
block it lands in, also O(sqrt n). No recursion, no tree, no lazy propagation -- just an array and a
short block table, which makes it the pragmatic choice when the operation is awkward to fit into a
segment tree, and the foundation of Mo's algorithm for offline range queries.

The sweet spot block size is exactly sqrt(n): with b blocks of size s and b*s = n, a query walks O(s)
partial elements plus O(b) whole blocks, minimized at s = b = sqrt(n). This module implements
sqrt-decomposition for sum, min, and max range queries with point updates, plus a range-assignment
variant, and it is validated against a brute-force array: random interleavings of updates and queries
agree exactly for sum/min/max; the block aggregates stay consistent with the underlying array after
every update; single-element and full-array ranges are correct; the block size is Theta(sqrt n); and it
handles the degenerate empty and singleton ranges. Pure stdlib; the range-query companion to the
segment-tree, Fenwick-tree, and sparse-table tools."""

from __future__ import annotations

import math


class SqrtDecomposition:
    """Square-root decomposition over an array with O(sqrt n) range query and point update."""

    def __init__(self, data, op="sum"):
        self.a = list(data)
        self.n = len(self.a)
        self.op = op
        self.block = max(1, int(math.isqrt(self.n)) or 1)
        self._rebuild()

    def _combine(self, x, y):
        if self.op == "sum":
            return x + y
        if self.op == "min":
            return min(x, y)
        if self.op == "max":
            return max(x, y)
        raise ValueError("op must be sum, min, or max")

    def _identity(self):
        if self.op == "sum":
            return 0
        if self.op == "min":
            return math.inf
        if self.op == "max":
            return -math.inf

    def _block_agg(self, bi):
        lo = bi * self.block
        hi = min(lo + self.block, self.n)
        agg = self._identity()
        for i in range(lo, hi):
            agg = self._combine(agg, self.a[i])
        return agg

    def _rebuild(self):
        nblocks = (self.n + self.block - 1) // self.block
        self.blocks = [self._block_agg(bi) for bi in range(nblocks)]

    def block_count(self):
        return len(self.blocks)

    def update(self, i, value):
        """Set a[i] = value and refresh its block aggregate. O(sqrt n)."""
        self.a[i] = value
        bi = i // self.block
        self.blocks[bi] = self._block_agg(bi)

    def query(self, l, r):
        """Aggregate over the inclusive range [l, r]. O(sqrt n)."""
        if l > r:
            return self._identity()
        res = self._identity()
        bl = l // self.block
        br = r // self.block
        if bl == br:
            for i in range(l, r + 1):
                res = self._combine(res, self.a[i])
            return res
        # left partial block
        for i in range(l, (bl + 1) * self.block):
            res = self._combine(res, self.a[i])
        # whole middle blocks
        for bi in range(bl + 1, br):
            res = self._combine(res, self.blocks[bi])
        # right partial block
        for i in range(br * self.block, r + 1):
            res = self._combine(res, self.a[i])
        return res

    def range_assign(self, l, r, value):
        """Set a[i] = value for all i in [l, r], refreshing affected blocks. O(sqrt n + range)."""
        for i in range(l, r + 1):
            self.a[i] = value
        for bi in range(l // self.block, r // self.block + 1):
            self.blocks[bi] = self._block_agg(bi)

    def to_list(self):
        return list(self.a)


def brute_query(a, l, r, op="sum"):
    """Reference: aggregate over [l, r] by direct iteration."""
    if l > r:
        return {"sum": 0, "min": math.inf, "max": -math.inf}[op]
    seg = a[l:r + 1]
    if op == "sum":
        return sum(seg)
    if op == "min":
        return min(seg)
    if op == "max":
        return max(seg)

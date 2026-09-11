"""Fenwick tree (binary indexed tree): running sums that update in log time.

Keep an array of numbers and you often want two things at once: change one element, and ask for
the sum of a prefix (or any range). A plain array gives instant updates but O(n) prefix sums; a
prefix-sum array gives instant sums but O(n) updates. Peter Fenwick's 1994 binary indexed tree
does BOTH in O(log n), using the binary representation of the indices as an implicit tree.

The trick: store partial sums keyed by the LOW bit of each index. Index i is responsible for the
range ending at i whose length is the value of its lowest set bit, i & (-i). To read a prefix
sum, walk downward stripping the low bit each step (i -= i & -i); to update, walk upward adding
it (i += i & -i). Each walk touches only as many nodes as there are bits, so both operations are
O(log n) with a single array and no pointers -- the whole structure is n integers.

From prefix sums you get any range sum by subtraction, and because the cumulative sums are
monotone (for nonnegative values) you can binary-search the tree to find, e.g., the smallest
index whose prefix sum reaches a target -- an O(log n) "select" used for weighted sampling,
order statistics, and rank queries. Fenwick trees power competitive-programming range queries,
database index statistics, and streaming quantiles.

This module implements the tree with point update, prefix and range sums, point reads, and the
cumulative binary search, and checks every operation against a brute-force array. Pure stdlib;
the data-structure companion to the HyperLogLog and Bloom notes.
"""

from __future__ import annotations


class FenwickTree:
    """A Fenwick / binary indexed tree over n slots (0-indexed API, 1-indexed internally)."""

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("size must be nonnegative")
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed; tree[0] unused

    @classmethod
    def from_values(cls, values):
        """Build a tree from an initial list of values in O(n)."""
        vals = list(values)
        ft = cls(len(vals))
        # linear construction: add each value to its own node, then propagate to parent
        for i in range(1, ft.n + 1):
            ft.tree[i] += vals[i - 1]
            parent = i + (i & -i)
            if parent <= ft.n:
                ft.tree[parent] += ft.tree[i]
        return ft

    def update(self, index: int, delta):
        """Add delta to the element at index (0-based). O(log n)."""
        if not 0 <= index < self.n:
            raise IndexError("index out of range")
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, count: int):
        """Sum of the first `count` elements (indices 0..count-1). O(log n).
        prefix_sum(0) = 0, prefix_sum(n) = total."""
        if not 0 <= count <= self.n:
            raise IndexError("count out of range")
        i = count
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_sum(self, lo: int, hi: int):
        """Sum of elements in the half-open range [lo, hi). O(log n)."""
        if not 0 <= lo <= hi <= self.n:
            raise IndexError("invalid range")
        return self.prefix_sum(hi) - self.prefix_sum(lo)

    def get(self, index: int):
        """Value of a single element at index (0-based). O(log n)."""
        return self.range_sum(index, index + 1)

    def set(self, index: int, value):
        """Set the element at index to `value` (via a delta update). O(log n)."""
        self.update(index, value - self.get(index))

    def total(self):
        """Sum of all elements."""
        return self.prefix_sum(self.n)

    def find_prefix(self, target):
        """Smallest index k (0-based) such that prefix_sum(k+1) >= target, assuming all values
        are nonnegative (so prefix sums are monotone). Returns n if no prefix reaches target.
        O(log n) via binary lifting over the tree -- the 'select' operation."""
        pos = 0
        remaining = target
        # highest power of two <= n
        log = 1
        while (log << 1) <= self.n:
            log <<= 1
        k = log
        while k > 0:
            nxt = pos + k
            if nxt <= self.n and self.tree[nxt] < remaining:
                pos = nxt
                remaining -= self.tree[nxt]
            k >>= 1
        return pos  # number of elements consumed; the answer index is `pos` (0-based)

    def to_list(self):
        """Reconstruct the underlying value array (for inspection/testing)."""
        return [self.get(i) for i in range(self.n)]

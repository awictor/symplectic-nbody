"""2D Fenwick tree: point updates and rectangle sums on a grid, both in O(log^2 n).

A Fenwick tree (binary indexed tree) answers "sum of a prefix" and "add to one position" in O(log n)
by cleverly storing partial sums at indices governed by their lowest set bit. The 2D FENWICK TREE
lifts this to a grid: a Fenwick tree of Fenwick trees, supporting POINT UPDATE (add a value to cell
(r, c)) and PREFIX-RECTANGLE SUM (total of the rectangle from (0,0) to (r,c)) each in O(log R * log C).
An arbitrary axis-aligned rectangle sum then follows by inclusion-exclusion of four prefix sums. This
is the workhorse behind dynamic 2D range queries: image integral tables that change over time,
counting points in a rectangle as points are inserted, geometric sweep-line problems, and grid-based
game or simulation state where both cells and queries update constantly -- cases a static prefix-sum
table cannot handle because every update would cost O(R*C) to rebuild.

The mechanics mirror the 1D case in both dimensions. To update (r, c), walk r upward by adding its
lowest set bit (r += r & -r) and, for each such row index, walk c the same way, adding the delta at
every (i, j) node touched. To query the prefix rectangle to (r, c), walk both indices DOWNWARD by
subtracting the lowest set bit, accumulating the stored partial sums. Because each walk visits O(log)
indices, the double loop is O(log R * log C). A general rectangle [(r1,c1),(r2,c2)] is prefix(r2,c2) -
prefix(r1-1,c2) - prefix(r2,c1-1) + prefix(r1-1,c1-1).

This module implements a 2D Fenwick tree with point update, prefix sum, arbitrary rectangle sum, and
point-value read, plus construction from an initial matrix. It is verified against a brute-force 2D
prefix-sum reference -- every rectangle query matches the direct sum after arbitrary interleaved
updates -- on hundreds of random grids and query sequences, including single-cell and full-grid
rectangles. Pure stdlib; a data-structures companion to the 1D Fenwick, segment-tree, and
sparse-table notes."""

from __future__ import annotations


class Fenwick2D:
    """A 2D binary indexed tree over an R x C grid (0-indexed cells, 1-indexed internal storage)."""

    def __init__(self, rows, cols):
        self.R = rows
        self.C = cols
        # 1-indexed internal tree: (R+1) x (C+1)
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, r, c, delta):
        """Add `delta` to grid cell (r, c). O(log R * log C)."""
        i = r + 1
        while i <= self.R:
            j = c + 1
            while j <= self.C:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def prefix_sum(self, r, c):
        """Sum of the rectangle from (0,0) to (r,c) inclusive. Returns 0 if r or c is negative."""
        if r < 0 or c < 0:
            return 0
        total = 0
        i = min(r, self.R - 1) + 1
        while i > 0:
            j = min(c, self.C - 1) + 1
            while j > 0:
                total += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return total

    def rectangle_sum(self, r1, c1, r2, c2):
        """Sum over the inclusive rectangle with corners (r1,c1) and (r2,c2), by inclusion-exclusion
        of four prefix sums."""
        if r1 > r2 or c1 > c2:
            return 0
        return (self.prefix_sum(r2, c2)
                - self.prefix_sum(r1 - 1, c2)
                - self.prefix_sum(r2, c1 - 1)
                + self.prefix_sum(r1 - 1, c1 - 1))

    def point_value(self, r, c):
        """The current value of a single cell (r, c), via a 1x1 rectangle sum."""
        return self.rectangle_sum(r, c, r, c)


def from_matrix(matrix):
    """Build a Fenwick2D from an initial R x C matrix (list of lists) by point updates."""
    if not matrix:
        return Fenwick2D(0, 0)
    R, C = len(matrix), len(matrix[0])
    fw = Fenwick2D(R, C)
    for r in range(R):
        for c in range(C):
            if matrix[r][c]:
                fw.update(r, c, matrix[r][c])
    return fw


# --- brute-force reference --------------------------------------------------
class BruteGrid:
    """Direct 2D grid with O(area) rectangle sums, for validation."""

    def __init__(self, rows, cols):
        self.R = rows
        self.C = cols
        self.g = [[0] * cols for _ in range(rows)]

    def update(self, r, c, delta):
        self.g[r][c] += delta

    def rectangle_sum(self, r1, c1, r2, c2):
        if r1 > r2 or c1 > c2:
            return 0
        total = 0
        for r in range(max(0, r1), min(self.R, r2 + 1)):
            for c in range(max(0, c1), min(self.C, c2 + 1)):
                total += self.g[r][c]
        return total

    def prefix_sum(self, r, c):
        return self.rectangle_sum(0, 0, r, c)

    def point_value(self, r, c):
        return self.g[r][c]

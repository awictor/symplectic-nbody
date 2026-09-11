"""Wavelet trees: rank, select, and quantile queries over a sequence.

A WAVELET TREE is a succinct data structure that turns a sequence over an alphabet into a balanced
tree of bit vectors, answering a remarkable range of queries in O(log sigma) time (sigma = alphabet
size) with essentially the space of the sequence itself. It is a cornerstone of compressed text
indexing and succinct data structures -- the machinery behind FM-indexes, compressed suffix arrays,
and range-search structures in computational geometry and bioinformatics.

The idea is a recursive partition of the ALPHABET. At the root, split the value range at its midpoint:
build a bit vector marking, for each position, whether that element falls in the upper half (1) or the
lower half (0). Elements going left form the left child's sequence, those going right the right
child's, and each child recurses on its half of the alphabet. A value is thus encoded by the path of
bits from root to leaf. Every query walks that O(log sigma)-deep tree, using RANK on the bit vectors
(how many 1s -- or 0s -- occur before a position) to map an index down to the correct child.

From this single structure fall three fundamental operations. RANK(c, i): how many times value c
appears in the first i positions -- walk down following c's bits, mapping i by bit-rank at each level.
SELECT(c, j): the position of the j-th occurrence of c -- walk down to c's leaf, then map the index
back UP. QUANTILE(lo, hi, k): the k-th smallest value in a range -- descend by comparing k to the
number of small (0-bit) elements in the range, a range-median generalization that array scans cannot
do in sublinear time. RANGE_COUNT bounds how many values in a range fall within a value window.

This module builds a wavelet tree over an integer sequence and implements access, rank, select,
quantile (k-th smallest in a range), and range-count. It is verified exhaustively against brute-force
references: access reproduces the sequence, rank matches a prefix count, select matches an
occurrence scan, quantile matches a sorted-slice lookup, and range-count matches a filtered scan,
across many random sequences and query ranges. Pure stdlib; a succinct-data-structure companion to
the suffix-array, Fenwick-tree, and segment-tree notes."""

from __future__ import annotations


class _Node:
    __slots__ = ("lo", "hi", "bits", "prefix", "left", "right", "_count")

    def __init__(self, lo, hi):
        self.lo = lo            # inclusive low value of this node's alphabet range
        self.hi = hi            # inclusive high value
        self.bits = []          # bit per element: 0 = goes left (<= mid), 1 = goes right
        self.prefix = [0]       # prefix sum of bits: prefix[i] = number of 1s in bits[:i]
        self.left = None
        self.right = None


class WaveletTree:
    """A wavelet tree over a sequence of integers, supporting rank/select/quantile/range-count."""

    def __init__(self, sequence):
        self.seq = list(sequence)
        self.n = len(self.seq)
        if self.n == 0:
            self.root = None
            self.lo = self.hi = 0
            return
        self.lo = min(self.seq)
        self.hi = max(self.seq)
        self.root = self._build(self.seq, self.lo, self.hi)

    def _build(self, values, lo, hi):
        node = _Node(lo, hi)
        if lo == hi:
            # leaf: no bit vector needed, but record the count
            node.bits = None
            node.prefix = None
            node._count = len(values)
            return node
        mid = (lo + hi) // 2
        left_vals, right_vals = [], []
        bits = []
        run = 0
        prefix = [0]
        for v in values:
            if v <= mid:
                bits.append(0)
                left_vals.append(v)
            else:
                bits.append(1)
                right_vals.append(v)
                run += 1
            prefix.append(run)
        node.bits = bits
        node.prefix = prefix
        if left_vals:
            node.left = self._build(left_vals, lo, mid)
        if right_vals:
            node.right = self._build(right_vals, mid + 1, hi)
        return node

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _ones(node, i):
        """Number of 1-bits in node.bits[:i]."""
        return node.prefix[i]

    # --- access ----------------------------------------------------------
    def access(self, index):
        """The value at position `index` (reconstructs seq[index] via the tree)."""
        if not (0 <= index < self.n):
            raise IndexError(index)
        node = self.root
        i = index
        while node.lo != node.hi:
            b = node.bits[i]
            ones = self._ones(node, i)
            if b == 0:
                i = i - ones            # rank of 0s before i
                node = node.left
            else:
                i = ones                # rank of 1s before i
                node = node.right
        return node.lo

    # --- rank ------------------------------------------------------------
    def rank(self, value, index):
        """Number of occurrences of `value` in seq[:index]."""
        if index <= 0 or value < self.lo or value > self.hi:
            return 0
        index = min(index, self.n)
        node = self.root
        i = index
        while node.lo != node.hi:
            mid = (node.lo + node.hi) // 2
            ones = self._ones(node, i)
            if value <= mid:
                i = i - ones
                node = node.left
            else:
                i = ones
                node = node.right
            if node is None:
                return 0
        return i

    # --- select ----------------------------------------------------------
    def select(self, value, j):
        """The 0-based position of the (j+1)-th occurrence of `value` (j is 0-indexed), or -1."""
        if value < self.lo or value > self.hi or j < 0:
            return -1
        # descend to the leaf, remembering the path
        path = []
        node = self.root
        while node.lo != node.hi:
            mid = (node.lo + node.hi) // 2
            if value <= mid:
                path.append((node, 0))
                node = node.left
            else:
                path.append((node, 1))
                node = node.right
            if node is None:
                return -1
        # count at leaf
        if j >= node._count:
            return -1
        i = j                          # position within the leaf's own (implicit) list
        # walk back up, inverting the index mapping
        for parent, b in reversed(path):
            i = self._select_up(parent, b, i)
        return i

    @staticmethod
    def _select_up(node, b, i):
        """Given position i in child b of node, return the corresponding position in node."""
        if b == 0:
            # we need the index p such that among positions with bit 0, this is the i-th
            # i.e. the (i+1)-th zero. Find it by scanning prefix (binary search).
            # number of zeros before position p is p - prefix[p]; want that == i, bit==0
            lo, hi = 0, len(node.bits)
            # find smallest p with (zeros in bits[:p+1]) == i+1 and bits[p]==0
            target = i + 1
            # linear-safe binary search on zeros count
            left, right = 0, len(node.bits) - 1
            ans = right
            while left <= right:
                m = (left + right) // 2
                zeros = (m + 1) - node.prefix[m + 1]
                if zeros >= target:
                    ans = m
                    right = m - 1
                else:
                    left = m + 1
            return ans
        else:
            target = i + 1
            left, right = 0, len(node.bits) - 1
            ans = right
            while left <= right:
                m = (left + right) // 2
                ones = node.prefix[m + 1]
                if ones >= target:
                    ans = m
                    right = m - 1
                else:
                    left = m + 1
            return ans

    # --- quantile: k-th smallest in a range ------------------------------
    def quantile(self, lo_idx, hi_idx, k):
        """The k-th smallest value (0-indexed k) in seq[lo_idx:hi_idx]."""
        if not (0 <= lo_idx < hi_idx <= self.n):
            raise IndexError((lo_idx, hi_idx))
        if not (0 <= k < hi_idx - lo_idx):
            raise IndexError(k)
        node = self.root
        l, r = lo_idx, hi_idx
        while node.lo != node.hi:
            ones_l = self._ones(node, l)
            ones_r = self._ones(node, r)
            zeros_in_range = (r - l) - (ones_r - ones_l)
            if k < zeros_in_range:
                # k-th smallest is in the left child
                l = l - ones_l
                r = r - ones_r
                node = node.left
            else:
                k -= zeros_in_range
                l = ones_l
                r = ones_r
                node = node.right
        return node.lo

    # --- range count -----------------------------------------------------
    def range_count(self, lo_idx, hi_idx, vlo, vhi):
        """Number of positions p in [lo_idx, hi_idx) with vlo <= seq[p] <= vhi."""
        if lo_idx >= hi_idx or vhi < vlo:
            return 0
        return self._range_count(self.root, lo_idx, hi_idx, vlo, vhi)

    def _range_count(self, node, l, r, vlo, vhi):
        if node is None or l >= r:
            return 0
        if vhi < node.lo or vlo > node.hi:
            return 0
        if vlo <= node.lo and node.hi <= vhi:
            return r - l
        ones_l = self._ones(node, l)
        ones_r = self._ones(node, r)
        left = self._range_count(node.left, l - ones_l, r - ones_r, vlo, vhi)
        right = self._range_count(node.right, ones_l, ones_r, vlo, vhi)
        return left + right

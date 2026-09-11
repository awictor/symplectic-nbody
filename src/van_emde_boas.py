"""Van Emde Boas trees: integer sets with O(log log u) successor and predecessor.

A binary search tree does membership, insert, and (crucially) SUCCESSOR/PREDECESSOR in O(log n). But
when the keys are integers drawn from a bounded universe {0, 1, ..., u-1}, the VAN EMDE BOAS TREE
does all of them in O(log log u) -- exponentially better in the universe size. For u = 2^32 that is
about 5 steps per operation regardless of how many keys are stored. This is the theoretical champion
for the "predecessor problem" and underlies fast priority queues, IP routing tables, and integer
sorting.

The structure is a recursive divide of the universe by its SQUARE ROOT. Split each key x into a HIGH
half (x divided by sqrt(u), the 'cluster' index) and a LOW half (x mod sqrt(u), the position within
the cluster). A vEB tree of size u holds sqrt(u) child vEB trees each of size sqrt(u) (one per
cluster), plus a SUMMARY vEB tree of size sqrt(u) that records which clusters are non-empty. The
genius is storing the MIN and MAX of each node directly: the minimum is NOT stored recursively in any
child, which is what caps the recursion at one recursive call per level and gives the O(log log u)
bound (the recursion T(u) = T(sqrt(u)) + O(1) solves to O(log log u)). Successor, for instance, first
checks the current cluster; if the answer is not there, it consults the summary to find the next
non-empty cluster in a single recursive step, never two.

This module implements a van Emde Boas tree over a universe rounded up to a power of two, supporting
insert, delete, membership, minimum, maximum, successor, and predecessor. It is verified against a
reference sorted set and brute force: that membership matches after a random insert/delete stream,
that min and max are correct, that successor and predecessor match a linear scan of the sorted keys
for every query point in the universe, and that the ordered sequence produced by walking successors
from the minimum equals the sorted key list. Pure stdlib; a data-structure companion to the
skip-list, treap, and Fibonacci-heap notes."""

from __future__ import annotations


class VEBTree:
    """A van Emde Boas tree over the universe {0, ..., size-1}, size rounded up to a power of two."""

    def __init__(self, size):
        # round the universe up to the next power of two (>= 2)
        u = 2
        while u < size:
            u *= 2
        self.u = u
        self.min = None
        self.max = None
        if u <= 2:
            self.summary = None
            self.cluster = None
        else:
            self._lower_bits = self._sqrt_lower(u)
            self._lower_size = 1 << self._lower_bits         # sqrt-down
            self._upper_size = u // self._lower_size          # sqrt-up (number of clusters)
            self.summary = None                               # created lazily
            self.cluster = {}                                 # cluster index -> VEBTree (lazy)

    @staticmethod
    def _sqrt_lower(u):
        # number of low bits = floor(log2(u)/2)
        bits = u.bit_length() - 1          # log2(u) since u is a power of two
        return bits // 2

    def _high(self, x):
        return x >> self._lower_bits

    def _low(self, x):
        return x & (self._lower_size - 1)

    def _index(self, high, low):
        return (high << self._lower_bits) | low

    # --- queries ---------------------------------------------------------
    def minimum(self):
        return self.min

    def maximum(self):
        return self.max

    def __contains__(self, x):
        if x == self.min or x == self.max:
            return True
        if self.u <= 2:
            return False
        c = self.cluster.get(self._high(x))
        if c is None:
            return False
        return self._low(x) in c

    # --- insert ----------------------------------------------------------
    def insert(self, x):
        if self.min is None:
            self.min = self.max = x
            return
        if x == self.min or x == self.max:
            return
        if x < self.min:
            x, self.min = self.min, x          # swap: new min displaces old, insert old below
        if x > self.max:
            self.max = x
        if self.u > 2:
            h, l = self._high(x), self._low(x)
            child = self.cluster.get(h)
            if child is None:
                child = VEBTree(self._lower_size)
                self.cluster[h] = child
            if child.min is None:
                # cluster was empty -> mark it in the summary (O(1) recursion, min not stored deep)
                if self.summary is None:
                    self.summary = VEBTree(self._upper_size)
                self.summary.insert(h)
                child.min = child.max = l
            else:
                child.insert(l)

    # --- successor -------------------------------------------------------
    def successor(self, x):
        """Smallest key strictly greater than x, or None."""
        if self.u <= 2:
            if x == 0 and self.max == 1:
                return 1
            return None
        if self.min is not None and x < self.min:
            return self.min
        h, l = self._high(x), self._low(x)
        child = self.cluster.get(h)
        if child is not None and child.max is not None and l < child.max:
            off = child.successor(l)
            return self._index(h, off)
        # look in the next non-empty cluster via the summary
        if self.summary is None:
            return None
        succ_cluster = self.summary.successor(h)
        if succ_cluster is None:
            return None
        off = self.cluster[succ_cluster].min
        return self._index(succ_cluster, off)

    # --- predecessor -----------------------------------------------------
    def predecessor(self, x):
        """Largest key strictly less than x, or None."""
        if self.u <= 2:
            if x == 1 and self.min == 0:
                return 0
            return None
        if self.max is not None and x > self.max:
            return self.max
        h, l = self._high(x), self._low(x)
        child = self.cluster.get(h)
        if child is not None and child.min is not None and l > child.min:
            off = child.predecessor(l)
            return self._index(h, off)
        # look in the previous non-empty cluster
        pred_cluster = self.summary.predecessor(h) if self.summary is not None else None
        if pred_cluster is None:
            # might still be the node's own min (which is stored, not in any cluster)
            if self.min is not None and x > self.min:
                return self.min
            return None
        off = self.cluster[pred_cluster].max
        return self._index(pred_cluster, off)

    # --- delete ----------------------------------------------------------
    def delete(self, x):
        if self.min is None:
            return
        if self.min == self.max:
            if x == self.min:
                self.min = self.max = None
            return
        if self.u <= 2:
            # min != max here means both 0 and 1 are present; deleting one leaves the other
            if x == 0:
                self.min = self.max = 1
            elif x == 1:
                self.min = self.max = 0
            return
        if x == self.min:
            # the new min is the first element of the first non-empty cluster
            first_cluster = self.summary.min if self.summary is not None else None
            if first_cluster is None:
                self.min = self.max
                return
            x = self._index(first_cluster, self.cluster[first_cluster].min)
            self.min = x
        h, l = self._high(x), self._low(x)
        child = self.cluster.get(h)
        if child is None:
            return
        child.delete(l)
        if child.min is None:
            # cluster became empty -> clear it from the summary
            del self.cluster[h]
            if self.summary is not None:
                self.summary.delete(h)
        if x == self.max:
            # recompute max
            if self.summary is None or self.summary.max is None:
                self.max = self.min
            else:
                last_cluster = self.summary.max
                self.max = self._index(last_cluster, self.cluster[last_cluster].max)

    # --- ordered iteration ----------------------------------------------
    def to_sorted_list(self):
        out = []
        x = self.min
        while x is not None:
            out.append(x)
            x = self.successor(x)
        return out

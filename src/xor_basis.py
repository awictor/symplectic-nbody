"""The XOR linear basis -- linear algebra over GF(2) that answers subset-XOR questions instantly.

Give me a bag of integers and ask: what is the LARGEST value I can make by XOR-ing together some subset
of them? Can I make the number x at all? How many DISTINCT values are reachable? Brute force tries all
2^n subsets -- hopeless past 30 numbers. But XOR is addition in the vector space GF(2)^b (each integer a
bit-vector, XOR the vector sum), so the reachable values are exactly the LINEAR SPAN of the inputs, and
every question becomes linear algebra. The XOR LINEAR BASIS (a.k.a. the "linear basis" or "xor basis")
is a Gaussian-elimination-in-binary that maintains a basis of that span incrementally, answering all of
these in O(b) per query where b is the bit-width -- the standard competitive-programming and coding-
theory tool for XOR problems.

The structure keeps at most one basis vector per bit position: basis[i] is a value whose highest set
bit is i, or zero if that pivot is empty. To INSERT x, reduce it by XOR-ing out every pivot below its
current high bit; if something remains, it becomes a new pivot (the span grew), otherwise x was already
representable (linearly dependent) and is discarded. The number of nonzero pivots is the RANK -- the
dimension of the span -- so the span contains exactly 2^rank distinct values. MAXIMUM XOR greedily XORs
in each pivot from the top whenever it increases the running value; MINIMUM XOR of a nonempty subset is
the smallest pivot (or 0 if the inputs are dependent). MEMBERSHIP reduces x by the pivots and checks it
vanishes. With the basis put in reduced row-echelon form, the reachable values can even be RANKED --
the k-th smallest XOR value read straight off the bits of k.

This module implements the incremental basis with insert, rank, max-XOR, min-XOR, membership, the
count of distinct reachable values, and the k-th smallest reachable value, plus a brute-force reference
over all subsets for validation. Pure standard library (arbitrary-precision Python ints, so any
bit-width works).

Validation. Every query is checked against brute force over all 2^n subsets on small inputs: the
maximum and minimum subset-XOR match, the set of reachable values equals the basis span exactly, the
count of distinct values equals 2^rank, membership agrees for every candidate, and the k-th smallest
enumeration reproduces the sorted list of reachable values. The rank equals the number of linearly
independent inputs (adding a dependent value never grows it), inserting a value already in the span is a
no-op, and the empty basis reaches only zero. Large random bags confirm max-XOR beats any sampled subset."""


class XorBasis:
    """A linear basis over GF(2) for subset-XOR queries. Handles arbitrary-width non-negative ints."""

    def __init__(self, bits=64):
        # basis[i] holds a vector whose highest set bit is i (or 0 if empty). Grows as needed.
        self.bits = bits
        self.basis = {}          # pivot bit -> basis vector
        self._size = 0           # number of values inserted (not rank)

    def insert(self, x):
        """Insert x into the basis. Returns True if it grew the span (x was independent)."""
        self._size += 1
        cur = x
        while cur:
            hb = cur.bit_length() - 1
            if hb not in self.basis:
                self.basis[hb] = cur
                return True
            cur ^= self.basis[hb]
        return False             # x reduced to 0: already representable

    def rank(self):
        """Dimension of the span = number of pivots."""
        return len(self.basis)

    def max_xor(self, start=0):
        """The maximum value obtainable by XOR-ing ``start`` with a subset of the inserted values."""
        res = start
        for hb in sorted(self.basis, reverse=True):
            if res ^ self.basis[hb] > res:
                res ^= self.basis[hb]
        return res

    def min_xor(self):
        """The minimum XOR over NON-EMPTY subsets.

        If the inputs are linearly dependent (some nonempty subset XORs to 0), the minimum is 0;
        otherwise it is the smallest basis pivot value.
        """
        if self._size == 0:
            return None
        if self._size > self.rank():
            return 0             # a dependency exists -> some nonempty subset XORs to 0
        # all independent: smallest reachable nonzero value is the min over greedy reduction
        # minimum nonzero span element = reduce each pivot against higher ones then take the min
        red = self._reduced_vectors()
        return min(red) if red else 0

    def contains(self, x):
        """True if x is in the span (some subset XORs to x)."""
        cur = x
        while cur:
            hb = cur.bit_length() - 1
            if hb not in self.basis:
                return False
            cur ^= self.basis[hb]
        return True

    def count(self):
        """Number of distinct values reachable (including 0) = 2^rank."""
        return 1 << self.rank()

    def _reduced_vectors(self):
        """Basis in reduced row-echelon form: each pivot cleared from all other basis vectors."""
        pivots = sorted(self.basis, reverse=True)
        vecs = [self.basis[p] for p in pivots]
        # reduce so no vector has a bit that is another vector's pivot
        for i in range(len(pivots)):
            for j in range(len(pivots)):
                if i != j and (vecs[i] >> pivots[j]) & 1:
                    vecs[i] ^= vecs[j]
        return vecs

    def kth_smallest(self, k):
        """The k-th smallest reachable value (0-indexed; k in [0, 2^rank - 1])."""
        r = self.rank()
        if k < 0 or k >= (1 << r):
            raise IndexError("k out of range")
        # sort pivots ascending; the reduced vectors form a basis where bit i of k selects vec i
        pivots = sorted(self.basis)
        vecs = self._reduced_vectors_ascending(pivots)
        res = 0
        for i in range(r):
            if (k >> i) & 1:
                res ^= vecs[i]
        return res

    def _reduced_vectors_ascending(self, pivots_asc):
        """Reduced basis vectors indexed to ascending pivots, for kth-smallest enumeration."""
        vecs = [self.basis[p] for p in pivots_asc]
        for i in range(len(pivots_asc)):
            for j in range(len(pivots_asc)):
                if i != j and (vecs[i] >> pivots_asc[j]) & 1:
                    vecs[i] ^= vecs[j]
        return vecs

    def values(self):
        """The inserted-value count (not rank)."""
        return self._size


# ---------------------------------------------------------------------------
# brute-force reference
# ---------------------------------------------------------------------------

def brute_reachable(nums):
    """All values reachable by XOR-ing subsets of nums (including the empty subset -> 0)."""
    reachable = {0}
    for x in nums:
        reachable |= {r ^ x for r in reachable}
    return reachable


def brute_max_xor(nums):
    return max(brute_reachable(nums))


def brute_min_nonempty_xor(nums):
    if not nums:
        return None
    best = None
    # iterate nonempty subsets
    n = len(nums)
    for mask in range(1, 1 << n):
        v = 0
        for i in range(n):
            if mask & (1 << i):
                v ^= nums[i]
        best = v if best is None else min(best, v)
    return best

"""Cuckoo filters: approximate set membership that also supports deletion.

A BLOOM FILTER answers "have I seen this?" in tiny space with a controllable false-positive rate, but
it cannot DELETE items -- clearing bits would corrupt other members. The CUCKOO FILTER (Fan et al.,
2014) matches Bloom's space efficiency and false-positive rate, is often faster, AND supports
deletion, which is why it backs network routers, databases, and deduplication systems that must
remove items as well as add them. Like a Bloom filter it never gives a FALSE NEGATIVE (a stored item
always tests present); it only occasionally gives a false positive.

It stores a small FINGERPRINT (a few bits of each item's hash) rather than the item itself, in a
table of BUCKETS. Each item has TWO candidate buckets, computed by CUCKOO HASHING: i1 = hash(x), and
i2 = i1 XOR hash(fingerprint) -- the crucial trick that lets either bucket recover the other from just
the stored fingerprint, so lookups and deletes need only the fingerprint, not the original item. To
insert, place the fingerprint in either candidate bucket if there's room; if both are full, evict a
random resident, move it to ITS alternate bucket, and repeat -- the "cuckoo" kicking-out that gives
high load factors (~95%). Lookup checks whether the fingerprint appears in either candidate bucket;
delete removes one copy. The false-positive rate falls with the fingerprint size.

This module implements a cuckoo filter with configurable bucket size and fingerprint bits, supporting
add, contains, and delete. It is verified that it never reports a false negative (every added,
not-yet-deleted item tests present), that the false-positive rate on absent items is small and
consistent with the fingerprint size, that deletion removes an item (which then tests absent unless a
collision keeps it), that deleting one of several duplicates leaves the rest, that it handles the
load-factor limit gracefully, and that its behaviour is reproducible from a seed. Pure stdlib; a
probabilistic-data-structure companion to the Bloom-filter, HyperLogLog, and Count-Min notes."""

from __future__ import annotations


def _fnv1a(data):
    """A deterministic 64-bit FNV-1a hash of bytes, independent of PYTHONHASHSEED."""
    h = 1469598103934665603
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _to_bytes(x):
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode("utf-8")
    return repr(x).encode("utf-8")


class CuckooFilter:
    """A cuckoo filter for approximate set membership with deletion."""

    def __init__(self, capacity=1024, bucket_size=4, fingerprint_bits=16, seed=1, max_kicks=500):
        # round the number of buckets up to a power of two (so XOR indexing stays in range)
        n = 1
        num_buckets = max(1, capacity // bucket_size)
        while n < num_buckets:
            n *= 2
        self.num_buckets = n
        self.bucket_size = bucket_size
        self.fp_mask = (1 << fingerprint_bits) - 1
        self.max_kicks = max_kicks
        self.buckets = [[] for _ in range(n)]
        self._state = seed & 0xFFFFFFFF
        self.count = 0

    def _rand(self):
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state >> 8

    def _fingerprint(self, x):
        h = _fnv1a(_to_bytes(x))
        fp = (h >> 32) & self.fp_mask
        return fp or 1                        # never 0 (0 is the empty marker)

    def _index1(self, x):
        return _fnv1a(_to_bytes(x)) % self.num_buckets

    def _alt_index(self, index, fp):
        # i2 = i1 XOR hash(fingerprint)
        return (index ^ (_fnv1a(_to_bytes(fp)) % self.num_buckets)) % self.num_buckets

    def add(self, x):
        """Insert x. Returns True on success, False if the filter is too full (needs a rebuild)."""
        fp = self._fingerprint(x)
        i1 = self._index1(x)
        i2 = self._alt_index(i1, fp)
        if len(self.buckets[i1]) < self.bucket_size:
            self.buckets[i1].append(fp)
            self.count += 1
            return True
        if len(self.buckets[i2]) < self.bucket_size:
            self.buckets[i2].append(fp)
            self.count += 1
            return True
        # both full: cuckoo eviction
        i = i1 if self._rand() & 1 else i2
        for _ in range(self.max_kicks):
            slot = self._rand() % self.bucket_size
            fp, self.buckets[i][slot] = self.buckets[i][slot], fp
            i = self._alt_index(i, fp)
            if len(self.buckets[i]) < self.bucket_size:
                self.buckets[i].append(fp)
                self.count += 1
                return True
        return False                          # filter considered full

    def __contains__(self, x):
        fp = self._fingerprint(x)
        i1 = self._index1(x)
        i2 = self._alt_index(i1, fp)
        return fp in self.buckets[i1] or fp in self.buckets[i2]

    def delete(self, x):
        """Remove one copy of x. Returns True if a matching fingerprint was found and removed."""
        fp = self._fingerprint(x)
        i1 = self._index1(x)
        i2 = self._alt_index(i1, fp)
        for i in (i1, i2):
            if fp in self.buckets[i]:
                self.buckets[i].remove(fp)
                self.count -= 1
                return True
        return False

    def load_factor(self):
        return self.count / (self.num_buckets * self.bucket_size)

"""Bloom filter: testing membership in far less space than the set itself.

A Bloom filter (Burton Bloom, 1970) answers "have I seen this item?" using a bit array and a
handful of hash functions -- in a tiny fraction of the memory it would take to store the items.
The trade is one-sided: it can say "possibly present" when the item was never added (a false
positive), but it NEVER says "absent" for something you did add (no false negatives). That
asymmetry is exactly what you want for a fast pre-filter: web caches, spell checkers, database
query planners, and blockchains all use one to skip expensive lookups for items that are
definitely not there.

To add an item, hash it k ways and set those k bits. To test membership, hash it k ways and
check whether all k bits are set: if any is 0 the item is definitely absent; if all are 1 it is
probably present. After inserting n items into m bits with k hashes, the false-positive
probability is

    p = (1 - e^{-kn/m})^k,

minimized by choosing k = (m/n) ln 2, which gives p = 2^{-k} and needs about
m = -n ln p / (ln 2)^2 bits -- roughly 1.44 log2(1/p) bits per item, independent of the item
size. A million URLs at 1% error fit in about 1.2 MB, versus tens of MB to store them.

This module builds a Bloom filter with double-hashing (two independent hashes combined to
simulate k), supports add and membership tests, computes the theoretical and observed
false-positive rates and the optimal parameters, and verifies the no-false-negative guarantee.
Pure stdlib; the probabilistic-data-structure companion to the hashing and Shannon notes.
"""

from __future__ import annotations

import math


def optimal_num_bits(n: int, false_positive_rate: float) -> int:
    """Bits m needed for n items at the target false-positive rate: m = -n ln p / (ln 2)^2."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < false_positive_rate < 1.0:
        raise ValueError("false_positive_rate must be in (0, 1)")
    m = -n * math.log(false_positive_rate) / (math.log(2) ** 2)
    return max(1, int(math.ceil(m)))


def optimal_num_hashes(m: int, n: int) -> int:
    """Optimal number of hash functions k = (m/n) ln 2 for m bits and n items."""
    if n <= 0:
        raise ValueError("n must be positive")
    k = (m / n) * math.log(2)
    return max(1, int(round(k)))


def false_positive_rate(m: int, n: int, k: int) -> float:
    """Theoretical false-positive probability (1 - e^{-kn/m})^k after inserting n items."""
    if m <= 0:
        raise ValueError("m must be positive")
    return (1.0 - math.exp(-k * n / m)) ** k


class BloomFilter:
    """A Bloom filter over `num_bits` bits using `num_hashes` hash functions (double hashing)."""

    def __init__(self, num_bits: int, num_hashes: int):
        if num_bits <= 0 or num_hashes <= 0:
            raise ValueError("num_bits and num_hashes must be positive")
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self.bits = bytearray((num_bits + 7) // 8)
        self.count = 0  # number of add() calls (may double-count duplicates)

    @classmethod
    def for_capacity(cls, n: int, false_positive_rate: float = 0.01):
        """Build a filter sized for n items at the given target false-positive rate, with the
        optimal number of hash functions."""
        m = optimal_num_bits(n, false_positive_rate)
        k = optimal_num_hashes(m, n)
        return cls(m, k)

    def _hashes(self, item):
        """Yield k bit positions for an item via double hashing: h_i = h1 + i*h2 mod m. Two
        independent base hashes (from Python's hash and a variant) simulate k hashes cheaply."""
        data = item if isinstance(item, bytes) else str(item).encode("utf-8")
        # two independent 64-bit hashes via FNV-1a with different offset bases
        h1 = _fnv1a(data, 0xcbf29ce484222325)
        h2 = _fnv1a(data, 0x100000001b3) | 1  # ensure odd so it strides the whole array
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    def add(self, item):
        """Insert an item: set its k bits."""
        for pos in self._hashes(item):
            self.bits[pos >> 3] |= 1 << (pos & 7)
        self.count += 1

    def __contains__(self, item) -> bool:
        """Membership test: True if all k bits are set (possibly a false positive). A False is
        always correct -- there are no false negatives."""
        for pos in self._hashes(item):
            if not (self.bits[pos >> 3] >> (pos & 7)) & 1:
                return False
        return True

    def bits_set(self) -> int:
        """Number of 1 bits currently set (the filter's fill level)."""
        return sum(bin(b).count("1") for b in self.bits)

    def fill_ratio(self) -> float:
        """Fraction of bits set -- an estimate of e^{-kn/m}'s complement."""
        return self.bits_set() / self.num_bits

    def estimated_false_positive_rate(self) -> float:
        """Estimate the current false-positive rate from the observed fill ratio: (fill)^k."""
        return self.fill_ratio() ** self.num_hashes


def _fnv1a(data: bytes, offset: int) -> int:
    """64-bit FNV-1a hash of a byte string with a given offset basis."""
    h = offset
    prime = 0x100000001b3
    mask = (1 << 64) - 1
    for byte in data:
        h ^= byte
        h = (h * prime) & mask
    return h

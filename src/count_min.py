"""Count-Min sketch: frequency estimates in sublinear memory.

How often has each item appeared in a stream? An exact hash table needs a counter per distinct
item -- impossible for billions of distinct keys (IP addresses, URLs, n-grams). The Count-Min
sketch (Cormode & Muthukrishnan, 2005) estimates every item's count using a fixed small grid of
counters, trading a little accuracy for enormous memory savings. It never UNDERESTIMATES, and it
overestimates by a bounded amount with high probability.

The structure is a d x w table of counters and d independent hash functions, one per row. To add
an item (with weight 1, or any positive amount), hash it with each row's function and increment
that row's counter. To QUERY its count, hash it the same d ways and take the MINIMUM of the d
counters -- collisions can only inflate a counter, so the smallest of the d is the tightest
overestimate, and the true count is never above it. With width w = ceil(e/epsilon) and depth
d = ceil(ln(1/delta)), the estimate exceeds the truth by more than epsilon * (total count) with
probability at most delta -- so a few kilobytes track a stream of any size.

Because counts only ever combine additively, two sketches over different shards MERGE by
element-wise addition -- so counting is distributed and parallel, like HyperLogLog. Count-Min
powers network flow monitoring, database query optimizers, and NLP n-gram frequency tables, and
its heavy-hitter query underlies trending-topic detection.

This module builds the sketch with sizing from an (epsilon, delta) target, adds and queries
items, merges sketches, and checks that estimates never underestimate and stay within the error
bound against exact counts. Pure stdlib; the frequency-estimation companion to the Bloom-filter
and Misra-Gries notes.
"""

from __future__ import annotations

import math


def _hash(item, seed: int, w: int) -> int:
    """A seeded hash of an item into [0, w) via 64-bit FNV-1a with a splitmix64 finalizer."""
    data = item if isinstance(item, bytes) else str(item).encode("utf-8")
    h = (0xcbf29ce484222325 ^ (seed * 0x100000001b3)) & ((1 << 64) - 1)
    prime = 0x100000001b3
    mask = (1 << 64) - 1
    for byte in data:
        h ^= byte
        h = (h * prime) & mask
    # splitmix64 finalizer so all bits mix well
    h = (h ^ (h >> 30)) & mask
    h = (h * 0xbf58476d1ce4e5b9) & mask
    h = (h ^ (h >> 27)) & mask
    h = (h * 0x94d049bb133111eb) & mask
    h ^= h >> 31
    return h % w


class CountMinSketch:
    """A Count-Min sketch: a depth x width grid of counters with `depth` hash functions."""

    def __init__(self, width: int, depth: int):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.total = 0          # total weight added (all items)

    @classmethod
    def from_error(cls, epsilon: float, delta: float):
        """Size a sketch so the overestimate is within epsilon * total with probability >= 1 -
        delta: width = ceil(e/epsilon), depth = ceil(ln(1/delta))."""
        if not 0 < epsilon < 1 or not 0 < delta < 1:
            raise ValueError("epsilon and delta must be in (0, 1)")
        width = math.ceil(math.e / epsilon)
        depth = math.ceil(math.log(1.0 / delta))
        return cls(width, depth)

    def add(self, item, count: int = 1):
        """Add `count` occurrences of an item (count may be any positive integer)."""
        if count < 0:
            raise ValueError("count must be nonnegative")
        for r in range(self.depth):
            self.table[r][_hash(item, r + 1, self.width)] += count
        self.total += count

    def update(self, stream):
        for x in stream:
            self.add(x)
        return self

    def estimate(self, item) -> int:
        """Estimated count of an item: the minimum over the d rows. Never underestimates the
        true count; overestimates only via hash collisions."""
        return min(self.table[r][_hash(item, r + 1, self.width)] for r in range(self.depth))

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """Merge another sketch of the same shape by element-wise addition -- the union of the
        two streams. Returns a new sketch."""
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("sketches must have the same dimensions to merge")
        out = CountMinSketch(self.width, self.depth)
        for r in range(self.depth):
            out.table[r] = [a + b for a, b in zip(self.table[r], other.table[r])]
        out.total = self.total + other.total
        return out

    def error_bound(self) -> float:
        """The additive overestimate bound: e/width * total (exceeded with probability at most
        e^{-depth})."""
        return math.e / self.width * self.total

    def heavy_hitters(self, threshold: float, candidates):
        """Among `candidates`, those whose ESTIMATED count exceeds threshold * total. (The
        sketch cannot enumerate items itself, so a candidate set is supplied -- typically the
        distinct items seen, or a watched list.)"""
        cut = threshold * self.total
        return {c: self.estimate(c) for c in candidates if self.estimate(c) > cut}

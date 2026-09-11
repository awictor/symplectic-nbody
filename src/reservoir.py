"""Reservoir sampling: a uniform sample from a stream of unknown length.

You are handed items one at a time -- log lines, tweets, sensor readings -- and must keep a
uniform random sample of k of them, but you do not know how many will arrive and cannot store
them all. Reservoir sampling (Vitter's Algorithm R, 1985; the idea is older) does it in a single
pass with O(k) memory, and every item that ever streams by ends up in the final sample with
exactly probability k/n -- no matter how large n turns out to be.

The trick: fill the reservoir with the first k items. Then for the i-th item (i > k, 1-indexed),
keep it with probability k/i, and if kept, evict a uniformly random existing sample element. A
short induction shows this keeps every seen item with probability exactly k/n at all times. For
k=1 it is the classic "pick a uniform random line from a file in one pass".

Two extensions are included. WEIGHTED reservoir sampling (Efraimidis-Spirakis A-Res) gives each
item i a key u_i^(1/w_i) with u_i uniform, and keeps the k largest keys -- so an item is sampled
with probability proportional to its weight. And the sample can be maintained INCREMENTALLY as
the stream grows, exposing the reservoir at any moment.

This module implements the unweighted and weighted samplers (seeded for reproducibility) and a
streaming reservoir object, and verifies the uniformity statistically with a chi-square test
over many independent runs. Pure stdlib; the streaming companion to the Misra-Gries and
HyperLogLog notes.
"""

from __future__ import annotations

import math


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def randint(self, k: int) -> int:
        """Uniform int in [0, k)."""
        return (self._next() >> 16) % k

    def random(self) -> float:
        """Uniform float in [0, 1)."""
        return (self._next() >> 8) / (1 << 24)


def sample(stream, k: int, seed: int = 1):
    """Algorithm R: return a uniform random sample of up to k items from `stream` (any iterable
    of unknown length) in one pass with O(k) memory. If the stream has fewer than k items, all
    are returned."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    rng = _Rng(seed)
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(i + 1)     # uniform in [0, i]
            if j < k:
                reservoir[j] = item
    return reservoir


def weighted_sample(items, weights, k: int, seed: int = 1):
    """Efraimidis-Spirakis A-Res weighted reservoir sampling: return k items sampled without
    replacement with probability proportional to their weights. Each item gets a key
    u^(1/w); the k items with the largest keys win."""
    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")
    if k <= 0:
        return []
    rng = _Rng(seed)
    keyed = []
    for item, w in zip(items, weights):
        u = rng.random()
        # key = u^(1/w); take logs to keep it stable: log key = (1/w) * log u
        log_key = math.log(u) / w if u > 0 else float("-inf")
        keyed.append((log_key, item))
    keyed.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in keyed[:k]]


class Reservoir:
    """A streaming reservoir that maintains a uniform k-sample as items are added one at a time."""

    def __init__(self, k: int, seed: int = 1):
        if k < 0:
            raise ValueError("k must be nonnegative")
        self.k = k
        self.rng = _Rng(seed)
        self.items = []
        self.n = 0          # total items seen

    def add(self, item):
        """Offer one item to the reservoir."""
        if self.n < self.k:
            self.items.append(item)
        else:
            j = self.rng.randint(self.n + 1)
            if j < self.k:
                self.items[j] = item
        self.n += 1

    def update(self, stream):
        for x in stream:
            self.add(x)
        return self

    def sample(self):
        """The current reservoir contents."""
        return list(self.items)


def selection_frequencies(n: int, k: int, trials: int, seed: int = 1):
    """Run reservoir sampling `trials` times over the stream range(n) and count how often each
    element ends up in the sample. Returns a list of n counts -- each should be ~ trials*k/n if
    the sampler is uniform. Used to check uniformity statistically."""
    counts = [0] * n
    for t in range(trials):
        for x in sample(range(n), k, seed=seed + t):
            counts[x] += 1
    return counts


def chi_square_uniformity(counts, expected) -> float:
    """Pearson chi-square statistic of observed selection counts against the uniform
    expectation -- small values mean the sampler looks uniform."""
    return sum((c - expected) ** 2 / expected for c in counts)

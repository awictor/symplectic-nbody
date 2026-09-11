"""The alias method: O(1) sampling from a discrete distribution.

To draw from a weighted die -- outcome i with probability p_i -- the obvious way is to build the
cumulative distribution and binary-search a random number, at O(log n) per draw. Walker's alias
method (1974, made simple by Vose in 1991) does it in O(1) per draw after an O(n) setup, no
matter how many outcomes there are. It is the standard for fast weighted sampling: loot tables,
Monte-Carlo species selection, particle spawning, and any hot loop that samples the same
categorical distribution millions of times.

The trick is to chop the distribution into n equal-area columns, each holding at most TWO
outcomes: a "main" outcome and an "alias". Scale every probability by n so the average column
has area 1. Repeatedly pair a "small" outcome (scaled prob < 1) with a "large" one (> 1): the
small one fills the rest of its column with a slice of the large one, which is then reduced and
re-sorted. To sample, pick a column uniformly (one integer roll), then flip a biased coin (one
float) to choose the column's main outcome or its alias. Two rolls, no search.

The result is exact: the long-run frequency of each outcome equals its weight. This module builds
the alias table by Vose's algorithm, samples from it (seeded for reproducibility), and verifies
the empirical frequencies match the target weights with a chi-square test. Pure stdlib; the
weighted-sampling companion to the reservoir-sampling note.
"""

from __future__ import annotations


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


class AliasSampler:
    """Walker-Vose alias table for O(1) sampling from a discrete distribution."""

    def __init__(self, weights, seed: int = 1):
        if not weights:
            raise ValueError("need at least one weight")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be nonnegative")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.n = len(weights)
        self.probs = [w / total for w in weights]     # normalized target distribution
        self.rng = _Rng(seed)
        self._build(self.probs)

    def _build(self, probs):
        n = self.n
        self.prob = [0.0] * n      # per-column probability of taking the "main" outcome
        self.alias = [0] * n       # the column's alias outcome
        scaled = [p * n for p in probs]
        small, large = [], []
        for i, s in enumerate(scaled):
            (small if s < 1.0 else large).append(i)
        while small and large:
            l = small.pop()
            g = large.pop()
            self.prob[l] = scaled[l]
            self.alias[l] = g
            scaled[g] = (scaled[g] + scaled[l]) - 1.0   # give g the leftover of l's column
            (small if scaled[g] < 1.0 else large).append(g)
        # any leftovers (floating-point residue) are certain columns
        for g in large:
            self.prob[g] = 1.0
        for l in small:
            self.prob[l] = 1.0

    def sample(self) -> int:
        """Draw one outcome index in O(1): pick a column, then flip its biased coin."""
        col = self.rng.randint(self.n)
        return col if self.rng.random() < self.prob[col] else self.alias[col]

    def sample_many(self, k: int):
        """Draw k outcomes."""
        return [self.sample() for _ in range(k)]

    def probabilities(self):
        """The normalized target probabilities the table encodes."""
        return list(self.probs)


def empirical_distribution(sampler: AliasSampler, draws: int):
    """Sample `draws` times and return the observed frequency of each outcome (fractions
    summing to 1)."""
    counts = [0] * sampler.n
    for _ in range(draws):
        counts[sampler.sample()] += 1
    return [c / draws for c in counts]


def chi_square(sampler: AliasSampler, draws: int) -> float:
    """Pearson chi-square of the observed sample counts against the target distribution -- small
    values mean the sampler matches the weights."""
    counts = [0] * sampler.n
    for _ in range(draws):
        counts[sampler.sample()] += 1
    chi = 0.0
    for i in range(sampler.n):
        expected = draws * sampler.probs[i]
        if expected > 0:
            chi += (counts[i] - expected) ** 2 / expected
    return chi

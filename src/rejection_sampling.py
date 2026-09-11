"""Rejection sampling: drawing from any density you can only evaluate.

You have a probability density f(x) you can compute at any point, but no direct way to sample
from it -- it might be an unnormalized posterior, a physics distribution, or a hand-drawn shape.
Rejection sampling (von Neumann, 1951) turns "I can evaluate f" into "I can sample f" using a
simpler PROPOSAL density g(x) you CAN sample, and a constant M with f(x) <= M g(x) everywhere:

    1. draw a candidate x from g,
    2. draw u uniform in [0, 1],
    3. accept x if u <= f(x) / (M g(x)); otherwise reject and repeat.

Geometrically you are throwing darts uniformly under the envelope M g(x) and keeping only those
that land under f(x); the kept x-values are distributed exactly as f. It works even when f is
UNNORMALIZED (you only know it up to a constant), which is why it underlies Bayesian computation.

The price is efficiency: the acceptance probability is 1/M (or area(f)/area(Mg) for unnormalized
f), so a loose envelope wastes darts. The tightest constant is M = max_x f(x)/g(x). The special
case g = uniform on [a, b] with M = max f is the simple "box" rejection sampler, exact for any
bounded density on an interval.

This module implements box rejection sampling on an interval and general rejection sampling with
an arbitrary proposal, reports the empirical acceptance rate against the theoretical 1/M, and
verifies the sampled distribution matches the target by its moments and a histogram chi-square.
Pure stdlib (seeded LCG); the sampling companion to the Box-Muller and alias-method notes.
"""

from __future__ import annotations

import math


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        """Uniform float in [0, 1)."""
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()


def sample_interval(pdf, a: float, b: float, n: int, m: float = None, seed: int = 1):
    """Box rejection sampling: draw n values from `pdf` on [a, b] using a uniform proposal and a
    bound m >= max pdf on [a, b]. If m is None it is estimated by scanning the interval. `pdf`
    need not be normalized. Returns (samples, acceptance_rate)."""
    rng = _Rng(seed)
    if m is None:
        m = _estimate_max(pdf, a, b) * 1.01     # small headroom so the bound holds
    samples = []
    tries = 0
    while len(samples) < n:
        x = rng.uniform(a, b)
        u = rng.random()
        tries += 1
        if u * m <= pdf(x):
            samples.append(x)
    return samples, len(samples) / tries


def sample(pdf, proposal_sampler, proposal_pdf, m: float, n: int, seed: int = 1):
    """General rejection sampling with an arbitrary proposal. `proposal_sampler(rng)` returns a
    candidate from g, `proposal_pdf(x)` evaluates g, and m satisfies pdf(x) <= m*g(x). Returns
    (samples, acceptance_rate)."""
    if m <= 0:
        raise ValueError("m must be positive")
    rng = _Rng(seed)
    samples = []
    tries = 0
    while len(samples) < n:
        x = proposal_sampler(rng)
        u = rng.random()
        tries += 1
        if u * m * proposal_pdf(x) <= pdf(x):
            samples.append(x)
    return samples, len(samples) / tries


def _estimate_max(pdf, a: float, b: float, steps: int = 2000) -> float:
    """Estimate max pdf on [a, b] by dense sampling (for the box sampler's bound)."""
    return max(pdf(a + (b - a) * i / steps) for i in range(steps + 1))


def theoretical_acceptance(pdf, a: float, b: float, m: float = None) -> float:
    """The expected acceptance rate of the box sampler: area(pdf) / (m * (b - a)), by numerical
    integration of pdf over [a, b]."""
    if m is None:
        m = _estimate_max(pdf, a, b) * 1.01
    area = _integrate(pdf, a, b)
    return area / (m * (b - a))


def _integrate(f, a: float, b: float, steps: int = 4000) -> float:
    """Composite trapezoidal integral of f over [a, b]."""
    h = (b - a) / steps
    total = 0.5 * (f(a) + f(b))
    for i in range(1, steps):
        total += f(a + i * h)
    return total * h


# --- descriptive statistics (for validation) -------------------------------

def mean(data) -> float:
    return sum(data) / len(data)


def variance(data) -> float:
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def histogram(data, a: float, b: float, bins: int):
    """Counts of `data` in `bins` equal buckets over [a, b]."""
    h = [0] * bins
    width = (b - a) / bins
    for x in data:
        if a <= x < b:
            h[min(bins - 1, int((x - a) / width))] += 1
    return h


def chi_square_fit(data, pdf, a: float, b: float, bins: int) -> float:
    """Chi-square of the sample histogram against the (normalized) target density over [a, b].
    Small values mean the samples match the target."""
    counts = histogram(data, a, b, bins)
    n = len(data)
    width = (b - a) / bins
    norm = _integrate(pdf, a, b)          # normalizing constant of pdf on [a, b]
    chi = 0.0
    for i, c in enumerate(counts):
        # expected count = n * (probability mass in this bin), integrating pdf over the bin
        # rather than a coarse midpoint sample (which biases chi-square on curved densities)
        lo = a + i * width
        mass = _integrate(pdf, lo, lo + width, steps=64) / norm
        expected = n * mass
        if expected > 0:
            chi += (c - expected) ** 2 / expected
    return chi

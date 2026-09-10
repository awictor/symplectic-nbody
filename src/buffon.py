"""Buffon's needle: estimating pi by dropping sticks.

Rule a floor with parallel lines a distance d apart and drop a needle of length L (L <= d) at
random. What is the chance it crosses a line? Buffon worked it out in 1777 -- the first
problem in geometric probability -- by integrating over the needle's random centre and angle:

    P(cross) = 2 L / (pi d).

The startling consequence: pi appears in a purely mechanical experiment. Drop N needles, count
the C that cross a line, and

    pi ~ 2 L N / (d C),

so tossing sticks estimates pi. It converges slowly (the error shrinks only as 1/sqrt(N), like
any Monte Carlo method) but it needs no measurement of pi anywhere -- pi emerges from counting
crossings. For the short-needle case the same integral gives the crossing probability directly;
for a "long" needle (L > d) the formula acquires an extra term, but the classic case is
L <= d.

This module gives the crossing probability, the pi estimate from a trial count, the number of
needles needed for a target accuracy (from the 1/sqrt(N) Monte Carlo error), and a direct
seeded simulation of needle drops, and reproduces the P = 2L/(pi d) probability and a
pi estimate near 3.14. Pure stdlib (seeded LCG using high bits, no random module); the
geometric-probability companion to the Polya and Monte-Carlo notes.
"""

from __future__ import annotations

import math


class _Rng:
    """Seeded LCG; uses high bits (low bits of these constants are non-random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def crossing_probability(length: float, spacing: float) -> float:
    """Probability a needle of length L crosses a line when dropped on a floor ruled with
    lines spacing d apart (short-needle case L <= d): P = 2 L / (pi d)."""
    if length > spacing:
        raise ValueError("short-needle formula requires L <= d")
    return 2.0 * length / (math.pi * spacing)


def estimate_pi(length: float, spacing: float, n_drops: int, n_cross: int) -> float:
    """Estimate pi from a Buffon experiment: pi ~ 2 L N / (d C), for N drops and C crossings.
    Returns inf if no crossings were counted."""
    if n_cross == 0:
        return float("inf")
    return 2.0 * length * n_drops / (spacing * n_cross)


def needles_for_accuracy(target_relative_error: float) -> int:
    """Rough number of drops for a target relative error in pi, from the 1/sqrt(N) Monte
    Carlo scaling: N ~ 1 / error^2. Slow convergence -- 1% needs ~1e4, 0.1% ~1e6."""
    return int(1.0 / (target_relative_error ** 2))


def simulate(length: float, spacing: float, n_drops: int, seed: int = 1):
    """Drop n_drops needles with a seeded RNG and return (n_cross, pi_estimate). Each needle's
    centre distance to the nearest line is uniform in [0, d/2] and its angle uniform in
    [0, pi/2); it crosses if (L/2) sin(angle) >= that distance."""
    rng = _Rng(seed)
    half = spacing / 2.0
    crosses = 0
    for _ in range(n_drops):
        dist = rng.random() * half              # distance of centre to nearest line
        angle = rng.random() * (math.pi / 2.0)  # acute angle to the lines
        if (length / 2.0) * math.sin(angle) >= dist:
            crosses += 1
    return crosses, estimate_pi(length, spacing, n_drops, crosses)


def monte_carlo_error(n_drops: int) -> float:
    """Expected relative error scale of the pi estimate, ~ 1/sqrt(N): halving the error needs
    four times the drops."""
    return 1.0 / math.sqrt(n_drops)

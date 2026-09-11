"""The Galton board: how independent coin flips build a bell curve.

Sir Francis Galton's "bean machine" (1894) is a board studded with n staggered rows of pegs. A
bead dropped at the top bounces off one peg per row, going left or right with probability 1/2
each, and lands in one of n+1 slots at the bottom. Its final slot is just the number of
right-bounces in n independent coin flips -- so the slot occupancy is the binomial distribution

    P(slot k) = C(n, k) p^k (1-p)^(n-k),

and, because it is a sum of n independent +-1 steps, the central limit theorem makes that
histogram converge to a Gaussian of mean np and variance np(1-p) as n grows. The board is the
classic physical demonstration of the CLT: no bead is guided, yet thousands of them pile into a
smooth bell curve, and shifting the peg bias p slides and skews the pile exactly as the binomial
predicts.

This module gives the exact binomial slot probabilities and their mean, variance, and mode, the
normal (Gaussian) approximation the CLT guarantees, the total-variation distance between them
(shrinking like 1/sqrt(n)), and a seeded Monte-Carlo drop of beads through the pegs. It confirms
the histogram is binomial, that it tends to the Gaussian, and that a biased board shifts the
peak to np. Pure stdlib; the central-limit companion to the random-walk and birthday notes.
"""

from __future__ import annotations

import math


def slot_probabilities(rows: int, p: float = 0.5):
    """Exact probability of landing in each of the rows+1 slots: the binomial pmf
    C(rows, k) p^k (1-p)^(rows-k) for k = 0..rows."""
    if rows < 0:
        raise ValueError("rows must be >= 0")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    q = 1.0 - p
    return [math.comb(rows, k) * p ** k * q ** (rows - k) for k in range(rows + 1)]


def mean_slot(rows: int, p: float = 0.5) -> float:
    """Mean landing slot: n p."""
    return rows * p


def variance_slot(rows: int, p: float = 0.5) -> float:
    """Variance of the landing slot: n p (1-p)."""
    return rows * p * (1.0 - p)


def mode_slot(rows: int, p: float = 0.5) -> int:
    """Most likely slot: floor((n+1) p) (the binomial mode)."""
    return int((rows + 1) * p)


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Gaussian density N(mu, sigma^2) at x."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def normal_approx(rows: int, p: float = 0.5):
    """CLT normal approximation to the slot probabilities: N(np, np(1-p)) evaluated at each
    integer slot (a list of rows+1 densities, the de Moivre-Laplace limit of the binomial)."""
    mu = mean_slot(rows, p)
    sigma = math.sqrt(variance_slot(rows, p))
    if sigma == 0:
        # degenerate p = 0 or 1: all mass on one slot
        out = [0.0] * (rows + 1)
        out[int(mu)] = 1.0
        return out
    return [normal_pdf(k, mu, sigma) for k in range(rows + 1)]


def total_variation_distance(rows: int, p: float = 0.5) -> float:
    """Total-variation distance between the exact binomial slot distribution and its normal
    approximation (renormalized over the slots): half the sum of absolute differences. Shrinks
    like 1/sqrt(rows) -- the CLT convergence rate."""
    binom = slot_probabilities(rows, p)
    approx = normal_approx(rows, p)
    s = sum(approx)
    approx = [a / s for a in approx] if s > 0 else approx
    return 0.5 * sum(abs(b - a) for b, a in zip(binom, approx))


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def drop_bead(rng, rows: int, p: float = 0.5) -> int:
    """Drop one bead: sum of `rows` independent right-bounces (each with probability p).
    Returns the landing slot 0..rows."""
    slot = 0
    for _ in range(rows):
        if rng.random() < p:
            slot += 1
    return slot


def simulate(rows: int, beads: int = 20000, p: float = 0.5, seed: int = 1):
    """Monte-Carlo drop of `beads` beads through `rows` peg rows. Returns the slot histogram as
    a list of rows+1 frequencies (fractions summing to 1)."""
    rng = _Rng(seed)
    counts = [0] * (rows + 1)
    for _ in range(beads):
        counts[drop_bead(rng, rows, p)] += 1
    return [c / beads for c in counts]

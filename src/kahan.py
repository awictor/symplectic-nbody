"""Kahan summation: adding floats without losing the small ones.

Add a million small numbers to a running total in the naive way and the answer drifts: once the
total is large, each tiny addend has fewer bits of the mantissa left to land in, and the
low-order part is silently rounded away. Summing 0.1 ten million times the naive way is off by a
visible amount; averaging sensor data or accumulating a dot product, the error grows with the
count.

Kahan's compensated summation (1965) fixes this with a running CORRECTION term c that carries the
low-order bits lost on the previous addition:

    y = x - c            # bring in x, minus what we still owe
    t = sum + y          # the big addition (loses low bits of y)
    c = (t - sum) - y    # recover exactly what was lost
    sum = t

The error no longer grows with n; it stays bounded by a small constant times the machine epsilon,
as if the total were computed in roughly twice the precision. Neumaier's variant improves it
further for the case where the next addend is larger in magnitude than the running sum. Pairwise
(cascade) summation -- recursively splitting the array and summing halves -- is another route,
with O(log n) error growth and no correction term.

This module implements naive, Kahan, Neumaier, and pairwise summation, and a compensated dot
product and running-mean accumulator, and checks them against Python's exact math.fsum on
ill-conditioned inputs. Pure stdlib; the numerical-stability companion to the Welford note."""

from __future__ import annotations

import math


def naive_sum(values):
    """The straightforward running total -- accumulates rounding error with n."""
    total = 0.0
    for x in values:
        total += x
    return total


def kahan_sum(values):
    """Kahan compensated summation: carries a correction term for the low-order bits lost on
    each addition, keeping the error bounded independent of n."""
    total = 0.0
    c = 0.0             # the running compensation (lost low-order bits)
    for x in values:
        y = x - c
        t = total + y
        c = (t - total) - y     # algebraically 0, but in float it is the rounding error
        total = t
    return total


def neumaier_sum(values):
    """Neumaier's improved compensated summation: also correct when the next addend exceeds the
    running total in magnitude (where plain Kahan can lose the compensation)."""
    total = 0.0
    c = 0.0
    for x in values:
        t = total + x
        if abs(total) >= abs(x):
            c += (total - t) + x
        else:
            c += (x - t) + total
        total = t
    return total + c


def pairwise_sum(values):
    """Pairwise (cascade) summation: recursively sum halves. Error grows like O(log n) instead
    of O(n), with no correction term. Falls back to a direct sum for small blocks."""
    a = list(values)
    n = len(a)
    if n <= 128:
        return naive_sum(a)
    mid = n // 2
    return pairwise_sum(a[:mid]) + pairwise_sum(a[mid:])


def kahan_dot(u, v):
    """Compensated dot product sum(u_i * v_i) -- the products are summed with Kahan
    compensation, so long or ill-conditioned vectors stay accurate."""
    if len(u) != len(v):
        raise ValueError("vectors must have the same length")
    total = 0.0
    c = 0.0
    for a, b in zip(u, v):
        y = a * b - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


class KahanAccumulator:
    """A streaming sum with Kahan compensation; also exposes a numerically stable running mean."""

    def __init__(self):
        self.total = 0.0
        self.c = 0.0
        self.n = 0

    def add(self, x):
        y = x - self.c
        t = self.total + y
        self.c = (t - self.total) - y
        self.total = t
        self.n += 1
        return self

    def sum(self) -> float:
        return self.total

    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


def relative_error(approx: float, exact: float) -> float:
    """Relative error |approx - exact| / |exact| (absolute error if exact is 0)."""
    if exact == 0.0:
        return abs(approx)
    return abs(approx - exact) / abs(exact)

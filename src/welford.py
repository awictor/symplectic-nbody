"""Welford's algorithm: mean and variance in one stable pass.

The textbook variance formula, Var = (sum of x^2)/n - mean^2, is a numerical disaster: it
subtracts two large, nearly equal numbers, and for data with a big offset (temperatures around
1e6, timestamps, prices) catastrophic cancellation can make it return a NEGATIVE variance.
Welford's algorithm (1962) computes the mean and variance in a single pass, updating running
values as each datum arrives, and never forms those large intermediates -- so it is both online
(no need to store the data) and numerically stable.

The update, on seeing the k-th value x, keeps a running mean and the sum of squared deviations
M2 = sum (x_i - mean)^2:

    delta   = x - mean
    mean   += delta / k
    M2     += delta * (x - mean)      # uses the NEW mean, so the two deltas differ

Then variance = M2 / n (population) or M2 / (n-1) (sample). Higher moments extend the same idea
(Terriberry's update carries M3 and M4 for skewness and kurtosis). And two accumulators MERGE by
combining their counts, means, and M2 with a correction term -- so statistics over shards can be
computed in parallel and summed, exactly, like a monoid.

This module provides an online accumulator for mean, variance, standard deviation, skewness, and
kurtosis, plus a parallel merge, and verifies the results against a two-pass computation and
demonstrates the naive formula's instability on offset data. Pure stdlib; the streaming-statistics
companion to the reservoir-sampling and Misra-Gries notes.
"""

from __future__ import annotations

import math


class Welford:
    """Online accumulator for the running mean, variance, and higher moments."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0        # sum of squared deviations from the mean
        self.M3 = 0.0        # third central moment accumulator (for skewness)
        self.M4 = 0.0        # fourth central moment accumulator (for kurtosis)

    def add(self, x):
        """Incorporate one value in O(1), updating all running moments (Terriberry's update)."""
        x = float(x)
        n1 = self.n
        self.n += 1
        delta = x - self.mean
        delta_n = delta / self.n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1
        self.mean += delta_n
        self.M4 += (term1 * delta_n2 * (self.n * self.n - 3 * self.n + 3)
                    + 6 * delta_n2 * self.M2 - 4 * delta_n * self.M3)
        self.M3 += term1 * delta_n * (self.n - 2) - 3 * delta_n * self.M2
        self.M2 += term1

    def update(self, data):
        for x in data:
            self.add(x)
        return self

    def count(self) -> int:
        return self.n

    def variance(self, ddof: int = 0) -> float:
        """Variance. ddof=0 is the population variance (M2/n); ddof=1 is the sample variance
        (M2/(n-1), Bessel's correction)."""
        if self.n - ddof <= 0:
            return 0.0
        return self.M2 / (self.n - ddof)

    def std(self, ddof: int = 0) -> float:
        """Standard deviation."""
        return math.sqrt(self.variance(ddof))

    def skewness(self) -> float:
        """Sample skewness (third standardized moment); 0 for a symmetric distribution."""
        if self.n < 2 or self.M2 == 0:
            return 0.0
        return math.sqrt(self.n) * self.M3 / (self.M2 ** 1.5)

    def kurtosis(self) -> float:
        """Kurtosis (fourth standardized moment); 3 for a normal distribution."""
        if self.n < 2 or self.M2 == 0:
            return 0.0
        return self.n * self.M4 / (self.M2 * self.M2)

    def merge(self, other: "Welford") -> "Welford":
        """Combine two accumulators (e.g. from parallel shards) into a new one, exactly. Uses
        Chan's parallel variance formula with the M3/M4 correction terms."""
        out = Welford()
        na, nb = self.n, other.n
        n = na + nb
        if n == 0:
            return out
        delta = other.mean - self.mean
        d2 = delta * delta
        d3 = d2 * delta
        d4 = d2 * d2
        out.n = n
        out.mean = self.mean + delta * nb / n
        out.M2 = self.M2 + other.M2 + d2 * na * nb / n
        out.M3 = (self.M3 + other.M3 + d3 * na * nb * (na - nb) / (n * n)
                  + 3 * delta * (na * other.M2 - nb * self.M2) / n)
        out.M4 = (self.M4 + other.M4
                  + d4 * na * nb * (na * na - na * nb + nb * nb) / (n ** 3)
                  + 6 * d2 * (na * na * other.M2 + nb * nb * self.M2) / (n * n)
                  + 4 * delta * (na * other.M3 - nb * self.M3) / n)
        return out


# --- two-pass reference (for validation) -----------------------------------

def two_pass_mean_variance(data):
    """Exact mean and population variance by an explicit two-pass computation."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    m = sum(data) / n
    var = sum((x - m) ** 2 for x in data) / n
    return m, var


def naive_variance(data):
    """The UNSTABLE naive variance E[x^2] - E[x]^2 -- can go negative on offset data. Included
    only to demonstrate the instability Welford avoids."""
    n = len(data)
    if n == 0:
        return 0.0
    mean_sq = sum(x * x for x in data) / n
    sq_mean = (sum(data) / n) ** 2
    return mean_sq - sq_mean

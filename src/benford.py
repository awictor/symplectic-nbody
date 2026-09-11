"""Benford's law: why leading digits are not uniform.

Count the leading (most significant) digit of the street addresses in a phone book, the areas
of the world's rivers, physical constants, stock prices, or the file sizes on a disk, and you
do not find each of 1-9 appearing a ninth of the time. Instead 1 leads about 30% of the time
and 9 barely 4.6%, following

    P(d) = log10(1 + 1/d),      d = 1..9,

Benford's law (Newcomb 1881, Benford 1938). The reason is scale invariance: a quantity that
spans many orders of magnitude is effectively uniform in its logarithm, and a uniform mantissa
in log space maps to this logarithmic digit law -- so the distribution is the only one
invariant under a change of units (inches to metres cannot change which digit leads). Data that
grow multiplicatively (Fibonacci numbers, powers of a constant, factorials, populations)
famously obey it, and departures from it flag fabricated accounting figures and election
returns, which is why forensic auditors test for it.

This module gives the exact first-digit and general-position digit probabilities, tallies the
leading-digit histogram of a dataset, and scores the fit with a chi-square statistic and the
total-variation distance. It confirms that Fibonacci numbers, powers of 2, and factorials
follow the law while a uniform sample does not. Pure stdlib; the digit-statistics companion to
the random-walk and percolation notes.
"""

from __future__ import annotations

import math


def leading_digit(x: float) -> int:
    """First significant (base-10) digit of |x|, 1..9. Zero has no leading digit -> 0."""
    x = abs(x)
    if x == 0:
        return 0
    # Work from the fractional part of log10 so arbitrarily large ints (factorials) never
    # overflow a float: 10^frac lands in [1, 10) and its integer part is the leading digit.
    lg = math.log10(x)
    # +1e-9 absorbs the log10/pow round-trip error that can land, e.g., 30 on 2.9999...
    fd = int(10.0 ** (lg - math.floor(lg)) + 1e-9)
    # guard floating error at the decade edge (e.g. 0.999999 -> should be 1)
    if fd == 0:
        fd = 1
    elif fd > 9:
        fd = 9
    return fd


def benford_probability(d: int) -> float:
    """Benford first-digit probability P(d) = log10(1 + 1/d) for d in 1..9."""
    if not 1 <= d <= 9:
        raise ValueError("first digit must be 1..9")
    return math.log10(1.0 + 1.0 / d)


def benford_distribution():
    """The nine first-digit probabilities as a list indexed 0->digit1 .. 8->digit9."""
    return [benford_probability(d) for d in range(1, 10)]


def digit_probability(d: int, position: int = 1) -> float:
    """Probability that the digit in the given significant position is d.

    position=1 is the leading digit (d in 1..9); position>=2 covers d in 0..9 and tends
    quickly to the uniform 0.1 as the position increases.
    """
    if position == 1:
        return benford_probability(d)
    if not 0 <= d <= 9:
        raise ValueError("digit must be 0..9")
    lo = 10 ** (position - 2)
    hi = 10 ** (position - 1)
    return sum(math.log10(1.0 + 1.0 / (10 * k + d)) for k in range(lo, hi))


def leading_digit_counts(data):
    """Histogram of leading digits over an iterable of numbers; returns a list of 9 counts
    for digits 1..9 (values with leading digit 0, i.e. exact zeros, are skipped)."""
    counts = [0] * 9
    for x in data:
        fd = leading_digit(x)
        if 1 <= fd <= 9:
            counts[fd - 1] += 1
    return counts


def leading_digit_frequencies(data):
    """Observed leading-digit fractions (list of 9), summing to 1 over the non-zero values."""
    counts = leading_digit_counts(data)
    total = sum(counts)
    if total == 0:
        return [0.0] * 9
    return [c / total for c in counts]


def chi_square(data):
    """Pearson chi-square statistic comparing the leading-digit histogram to Benford.

    chi2 = sum_d (O_d - E_d)^2 / E_d, with 8 degrees of freedom. Small values (below the
    ~15.5 five-percent critical value for 8 dof) are consistent with Benford's law.
    """
    counts = leading_digit_counts(data)
    total = sum(counts)
    if total == 0:
        return 0.0
    chi2 = 0.0
    for i, d in enumerate(range(1, 10)):
        expected = total * benford_probability(d)
        chi2 += (counts[i] - expected) ** 2 / expected
    return chi2


def total_variation_distance(data):
    """Total-variation distance between the observed and Benford digit distributions:
    half the sum of absolute differences, in [0, 1]. 0 = perfect fit."""
    obs = leading_digit_frequencies(data)
    ben = benford_distribution()
    return 0.5 * sum(abs(o - b) for o, b in zip(obs, ben))


def follows_benford(data, alpha_critical: float = 15.51) -> bool:
    """True if the leading-digit distribution is consistent with Benford's law at the given
    chi-square critical value (default: 5% level, 8 degrees of freedom = 15.51)."""
    return chi_square(data) <= alpha_critical


# --- generators of classic Benford-obeying sequences (handy for demos/tests) ---

def fibonacci(n: int):
    """First n Fibonacci numbers (1, 1, 2, 3, 5, ...)."""
    seq = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


def powers(base: float, n: int):
    """base^0 .. base^(n-1)."""
    return [base ** k for k in range(n)]


def factorials(n: int):
    """1!, 2!, ..., n! (as exact ints)."""
    seq = []
    f = 1
    for k in range(1, n + 1):
        f *= k
        seq.append(f)
    return seq

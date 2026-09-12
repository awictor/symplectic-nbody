"""Kolmogorov-Smirnov test: comparing distributions by their largest gap.

The KOLMOGOROV-SMIRNOV test asks whether a sample comes from a given distribution (one-sample), or
whether two samples come from the same distribution (two-sample), WITHOUT assuming any particular
shape -- it is nonparametric, distribution-free. Its statistic is beautifully simple: the largest
vertical distance D between the two cumulative distribution functions being compared. For the
one-sample test that is the biggest gap between the sample's EMPIRICAL CDF (a staircase rising 1/n at
each data point) and the reference CDF; for the two-sample test it is the biggest gap between the two
empirical CDFs. Under the null hypothesis this D is small and its distribution is known independently
of the underlying distribution, which is the whole point -- one table of critical values works for any
continuous distribution. It underlies goodness-of-fit checking, A/B-test distribution comparison, and
random-number-generator quality testing.

The one-sample statistic is computed by sorting the sample and, at each point x_i, taking the maximum
of |i/n - F(x_i)| and |(i-1)/n - F(x_i)| (the ECDF jumps at x_i, so both the value just before and
just after must be checked against the reference F). The two-sample statistic sorts the merged
samples and tracks the two running ECDF fractions, taking the largest absolute difference. The
asymptotic P-VALUE comes from the Kolmogorov distribution: P(D > d) approx 2 * sum_{k>=1} (-1)^(k-1)
exp(-2 k^2 (sqrt(n) d)^2), a rapidly converging alternating series, using the effective sample size
sqrt(n) (one-sample) or sqrt(n m / (n+m)) (two-sample). A small p-value means the distributions differ.

This module computes the empirical CDF, the one- and two-sample KS statistics, and their asymptotic
p-values. It is verified against direct definitions -- the statistic equals the max ECDF gap computed
on a fine grid, it is symmetric in the two-sample case, it is invariant to relabelling, and it is zero
for identical inputs -- and statistically: samples drawn from the SAME distribution rarely reject at
5%, while samples from clearly DIFFERENT distributions reject with high power, over many seeded trials.
Pure stdlib; a statistics companion to the Welford, bootstrap, and Shannon-entropy notes."""

from __future__ import annotations

import math


def empirical_cdf(sample):
    """Return a function F(x) giving the fraction of `sample` values <= x (the empirical CDF)."""
    s = sorted(sample)
    n = len(s)
    import bisect

    def F(x):
        return bisect.bisect_right(s, x) / n if n else 0.0

    return F


def ks_one_sample(sample, cdf):
    """The one-sample KS statistic D = max_x |ECDF(x) - cdf(x)|, comparing `sample` to a reference
    `cdf` function. Returns D in [0, 1]."""
    s = sorted(sample)
    n = len(s)
    if n == 0:
        return 0.0
    d = 0.0
    for i, x in enumerate(s):
        fx = cdf(x)
        d = max(d, abs((i + 1) / n - fx), abs(i / n - fx))
    return d


def ks_two_sample(a, b):
    """The two-sample KS statistic D = max_x |ECDF_a(x) - ECDF_b(x)|. Returns D in [0, 1]."""
    sa = sorted(a)
    sb = sorted(b)
    na, nb = len(sa), len(sb)
    if na == 0 or nb == 0:
        return 0.0
    ia = ib = 0
    d = 0.0
    while ia < na and ib < nb:
        # advance past the smallest value, consuming its whole run in BOTH samples so that ties
        # (including identical samples) leave the two ECDFs level rather than momentarily apart
        x = sa[ia] if sa[ia] <= sb[ib] else sb[ib]
        while ia < na and sa[ia] == x:
            ia += 1
        while ib < nb and sb[ib] == x:
            ib += 1
        d = max(d, abs(ia / na - ib / nb))
    return d


def _kolmogorov_pvalue(d, effective_n):
    """P(D > d) under the null, from the asymptotic Kolmogorov distribution. `effective_n` is sqrt(n)
    for one-sample or sqrt(nm/(n+m)) for two-sample."""
    t = effective_n * d
    if t <= 0:
        return 1.0
    # Q(t) = 2 sum_{k=1}^inf (-1)^(k-1) exp(-2 k^2 t^2)
    s = 0.0
    for k in range(1, 101):
        term = ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * t * t)
        s += term
        if abs(term) < 1e-12:
            break
    p = 2.0 * s
    return max(0.0, min(1.0, p))


def ks_one_sample_test(sample, cdf):
    """One-sample KS test. Returns (D, p_value); small p means the sample does not fit the reference
    distribution."""
    n = len(sample)
    d = ks_one_sample(sample, cdf)
    return d, _kolmogorov_pvalue(d, math.sqrt(n))


def ks_two_sample_test(a, b):
    """Two-sample KS test. Returns (D, p_value); small p means the two samples differ."""
    na, nb = len(a), len(b)
    d = ks_two_sample(a, b)
    eff = math.sqrt(na * nb / (na + nb)) if (na + nb) else 0.0
    return d, _kolmogorov_pvalue(d, eff)


# --- standard reference CDFs (stdlib) --------------------------------------
def uniform_cdf(lo=0.0, hi=1.0):
    def F(x):
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0
        return (x - lo) / (hi - lo)
    return F


def normal_cdf(mu=0.0, sigma=1.0):
    def F(x):
        return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))
    return F


def exponential_cdf(rate=1.0):
    def F(x):
        return 0.0 if x < 0 else 1.0 - math.exp(-rate * x)
    return F


# --- brute-force reference --------------------------------------------------
def brute_ks_one_sample(sample, cdf, grid_points=2000):
    """The one-sample KS statistic by scanning |ECDF - cdf| on a fine grid over the sample range."""
    if not sample:
        return 0.0
    F = empirical_cdf(sample)
    lo, hi = min(sample), max(sample)
    span = hi - lo if hi > lo else 1.0
    d = 0.0
    for i in range(grid_points + 1):
        x = lo - 0.01 * span + (span * 1.02) * i / grid_points
        d = max(d, abs(F(x) - cdf(x)))
    # also sample exactly at data points (where the ECDF jumps)
    for x in sample:
        d = max(d, abs(F(x) - cdf(x)))
        d = max(d, abs((sorted(sample).index(x)) / len(sample) - cdf(x)))
    return d


def brute_ks_two_sample(a, b, grid_points=2000):
    """The two-sample KS statistic by scanning |ECDF_a - ECDF_b| on a fine grid."""
    if not a or not b:
        return 0.0
    Fa = empirical_cdf(a)
    Fb = empirical_cdf(b)
    allv = sorted(a) + sorted(b)
    lo, hi = min(allv), max(allv)
    span = hi - lo if hi > lo else 1.0
    d = 0.0
    for i in range(grid_points + 1):
        x = lo - 0.01 * span + (span * 1.02) * i / grid_points
        d = max(d, abs(Fa(x) - Fb(x)))
    for x in allv:
        d = max(d, abs(Fa(x) - Fb(x)))
    return d

"""Mann-Kendall trend test: is a time series monotonically trending, without assuming a shape or distribution?

Fitting a line and testing its slope assumes the trend is LINEAR and the noise NORMAL -- assumptions that
fail for skewed environmental data, and a few outliers can invent or hide a trend. The MANN-KENDALL TEST
(Mann 1945, Kendall 1975) is the nonparametric standard for detecting a MONOTONIC trend (increasing or
decreasing, of any shape) in a time series, used throughout hydrology, climatology, and air-quality
monitoring. It counts, over every pair of time points i < j, the sign of x_j - x_i:

    S = sum_{i<j} sign(x_j - x_i).

A strong upward trend makes most later values exceed earlier ones, driving S positive; a downward trend
drives it negative; no trend leaves S near zero. Under the null of no trend and independent data, S is
approximately normal with mean 0 and variance

    Var(S) = [ n(n-1)(2n+5) - sum_t t(t-1)(2t+5) ] / 18,

the second term correcting for groups of t tied values. A continuity-corrected z-score gives the p-value.
The trend MAGNITUDE is reported separately as the SEN SLOPE -- the median of all pairwise slopes
(x_j - x_i)/(j - i) -- which is robust to outliers and pairs naturally with the Mann-Kendall significance.

This module computes S, its tie-corrected variance, the z-score and p-value, Kendall's tau as a
normalized effect size, and the Sen slope (reusing the repo's Theil-Sen estimator). It is validated: a
clean increasing series gives large positive S and a tiny p-value, a decreasing one large negative S; a
trendless (shuffled) series gives S near zero and a large p-value; the test is invariant under any
monotonic transform of the values; ties inflate no false trend and are handled by the variance correction;
the Sen slope recovers the true slope of a noisy linear trend and matches Theil-Sen; a nonlinear but
monotone trend is still detected; and results are deterministic. Pure stdlib; the trend-detection
companion to the Theil-Sen, Kendall-tau, CUSUM, and PELT tools."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import theil_sen as _ts


def _s_statistic(x):
    """Mann-Kendall S = sum_{i<j} sign(x_j - x_i). O(n^2)."""
    n = len(x)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = x[j] - x[i]
            s += (d > 0) - (d < 0)
    return s


def _variance(x):
    """Tie-corrected variance of S under the null of no trend."""
    from collections import Counter
    n = len(x)
    base = n * (n - 1) * (2 * n + 5)
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in Counter(x).values())
    return (base - tie_term) / 18.0


def sen_slope(x, times=None):
    """Sen's slope: median of pairwise slopes (x_j - x_i)/(t_j - t_i). Robust trend magnitude.

    times defaults to 0,1,2,... (evenly sampled). Reuses the Theil-Sen estimator."""
    if times is None:
        times = list(range(len(x)))
    slope, _intercept = _ts.theil_sen(times, x)
    return slope


def mann_kendall(x, alternative="two-sided"):
    """Full Mann-Kendall trend test. Returns a dict with S, var_s, z, p_value, tau, sen_slope, and trend.

    alternative in {'two-sided','increasing','decreasing'}. tau = S / (n(n-1)/2) is the normalized
    effect size; trend is 'increasing'/'decreasing'/'no trend' at the 0.05 level (two-sided)."""
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 points")
    s = _s_statistic(x)
    var_s = _variance(x)

    # continuity-corrected z
    if var_s <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    def phi(t):
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))

    if alternative == "increasing":
        p = 1 - phi(z)
    elif alternative == "decreasing":
        p = phi(z)
    else:
        p = min(1.0, 2 * (1 - phi(abs(z))))

    tau = s / (n * (n - 1) / 2)
    slope = sen_slope(x)

    if p < 0.05 and alternative == "two-sided":
        trend = "increasing" if s > 0 else "decreasing"
    elif alternative == "increasing" and p < 0.05:
        trend = "increasing"
    elif alternative == "decreasing" and p < 0.05:
        trend = "decreasing"
    else:
        trend = "no trend"

    return {
        "S": s,
        "var_s": var_s,
        "z": z,
        "p_value": p,
        "tau": tau,
        "sen_slope": slope,
        "trend": trend,
    }

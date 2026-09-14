"""Kruskal-Wallis H test: nonparametric one-way ANOVA -- do several groups differ, without assuming normality?

One-way ANOVA tests whether several groups share a common mean, but it assumes normal, equal-variance data
and is fooled by outliers and skew. The KRUSKAL-WALLIS TEST (1952) is its rank-based, distribution-free
generalization -- the multi-group extension of the Mann-Whitney U test. It pools all observations, replaces
them by their RANKS, and asks whether the groups' average ranks differ more than chance would allow:

    H = 12 / (N(N+1)) * sum_g n_g * (Rbar_g - (N+1)/2)^2,

where N is the total sample size, n_g the size of group g, and Rbar_g its mean rank. Under the null of
identical distributions, H follows approximately a chi-squared distribution with k-1 degrees of freedom
(k groups), so a p-value comes from the chi-squared survival function. Ties are handled by average ranks
and a divisor correction 1 - sum(t^3 - t)/(N^3 - N). Because it uses only ranks, Kruskal-Wallis needs no
normality, resists outliers, and reduces exactly to the (squared, chi-squared-approximated) Mann-Whitney
test when k = 2.

This module computes H with tie correction, its chi-squared p-value (via a from-scratch lower-incomplete-
gamma routine), the mean rank of each group, and the epsilon-squared effect size. It is validated: on
groups drawn from one distribution H is small and the p-value large; on clearly shifted groups H is large
and the p-value tiny; for two groups H matches the Mann-Whitney normal-approximation z-squared; H is
invariant under any monotonic transform of the data; ties lower H via the correction; the chi-squared
survival function matches known values; the effect size rises with separation; and results are
deterministic. Pure stdlib; the nonparametric-ANOVA companion to the Mann-Whitney, KS-test, and
permutation-test tools."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mann_whitney import _average_ranks


def _lower_incomplete_gamma_reg(s, x):
    """Regularized lower incomplete gamma P(s, x) = gamma(s, x) / Gamma(s), via series or continued
    fraction. Used for the chi-squared CDF."""
    if x < 0 or s <= 0:
        return 0.0
    if x == 0:
        return 0.0
    if x < s + 1.0:
        # series expansion
        term = 1.0 / s
        total = term
        n = s
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    else:
        # continued fraction for the upper incomplete gamma Q, then P = 1 - Q
        tiny = 1e-300
        b = x + 1.0 - s
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, 1000):
            an = -i * (i - s)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        q = math.exp(-x + s * math.log(x) - math.lgamma(s)) * h
        return 1.0 - q


def chi2_sf(x, df):
    """Chi-squared survival function P(X > x) = 1 - CDF, for df degrees of freedom."""
    if x <= 0:
        return 1.0
    return 1.0 - _lower_incomplete_gamma_reg(df / 2.0, x / 2.0)


def h_statistic(groups):
    """Kruskal-Wallis H with tie correction. groups is a list of samples. Returns (H, mean_ranks)."""
    all_vals = []
    for g in groups:
        all_vals.extend(g)
    N = len(all_vals)
    ranks = _average_ranks(all_vals)
    # split ranks back per group
    idx = 0
    mean_ranks = []
    rank_sum_term = 0.0
    for g in groups:
        n_g = len(g)
        r = ranks[idx:idx + n_g]
        idx += n_g
        Rbar = sum(r) / n_g
        mean_ranks.append(Rbar)
        rank_sum_term += n_g * (Rbar - (N + 1) / 2.0) ** 2
    H = 12.0 / (N * (N + 1)) * rank_sum_term
    # tie correction
    from collections import Counter
    tie_term = sum(t ** 3 - t for t in Counter(all_vals).values())
    correction = 1.0 - tie_term / (N ** 3 - N) if N > 1 else 1.0
    if correction > 0:
        H /= correction
    return H, mean_ranks


def kruskal_wallis(groups):
    """Full Kruskal-Wallis test. Returns a dict with H, df, p_value, mean_ranks, and epsilon_squared."""
    k = len(groups)
    if k < 2:
        raise ValueError("need at least 2 groups")
    N = sum(len(g) for g in groups)
    H, mean_ranks = h_statistic(groups)
    df = k - 1
    p = chi2_sf(H, df)
    # epsilon-squared effect size: (H - k + 1) / (N - k)
    eps2 = (H - k + 1) / (N - k) if N > k else 0.0
    return {
        "H": H,
        "df": df,
        "p_value": p,
        "mean_ranks": mean_ranks,
        "epsilon_squared": max(0.0, eps2),
    }

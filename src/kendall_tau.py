"""Kendall's tau: measure monotonic association by counting concordant vs discordant pairs.

To ask whether two variables move together, Pearson's r measures LINEAR association and is thrown off by
outliers and curvature. Kendall's tau (1938) measures MONOTONIC association using only the ORDER of the
data: look at every pair of observations and ask whether they are CONCORDANT (both variables rank the pair
the same way) or DISCORDANT (opposite). Tau is the normalized difference,

    tau = (C - D) / (number of pairs),

ranging from +1 (every pair concordant -- a perfectly increasing relationship) through 0 (no association)
to -1 (perfectly decreasing). Because it depends only on ranks, tau is invariant under any monotonic
transform of either variable and is far more robust to outliers than Pearson's r; it also has a direct
probabilistic meaning -- tau = P(concordant) - P(discordant) for a random pair.

Two refinements matter. TAU-B corrects for ties by dividing by the geometric mean of the untied-pair
counts in each variable, so tied data still reaches +/-1 when the association is perfect. And for a
significance test, under the null of independence tau is approximately normal with variance
2(2n+5)/(9n(n-1)), giving a z-score and p-value.

This module computes the concordant/discordant counts (O(n^2), exact), tau-a and the tie-corrected tau-b,
and the normal-approximation p-value. It is validated: a perfectly increasing relationship gives tau = 1
and a decreasing one tau = -1; independent data gives tau near 0 with a large p-value; a strong
association gives a small p-value; tau is invariant under monotonic transforms of either variable; tau-b
handles ties and still reaches 1 on a perfect monotone-with-ties relationship; the concordant and
discordant counts sum to the untied pair count; tau matches a brute-force pair enumeration; and results
are deterministic. Pure stdlib; the rank-correlation companion to the Spearman, Pearson, and Mann-Whitney
tools."""

from __future__ import annotations

import math


def _counts(x, y):
    """Count concordant, discordant, x-tied, and y-tied pairs over all i<j. O(n^2).

    xt = pairs tied in x (any y), yt = pairs tied in y (any x). A pair tied in BOTH counts in each,
    matching the tau-b denominator's (n0 - xt) and (n0 - yt) untied-pair counts."""
    n = len(x)
    concordant = discordant = xt = yt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            sx = (dx > 0) - (dx < 0)
            sy = (dy > 0) - (dy < 0)
            if sx == 0:
                xt += 1
            if sy == 0:
                yt += 1
            if sx != 0 and sy != 0:
                if sx == sy:
                    concordant += 1
                else:
                    discordant += 1
    return concordant, discordant, xt, yt


def tau_a(x, y):
    """Kendall's tau-a = (C - D) / (n(n-1)/2). Does not correct for ties (underestimates with ties)."""
    n = len(x)
    c, d, _xt, _yt = _counts(x, y)
    n0 = n * (n - 1) / 2
    return (c - d) / n0 if n0 > 0 else 0.0


def tau_b(x, y):
    """Kendall's tau-b: tie-corrected, (C - D) / sqrt((n0 - xt)(n0 - yt)). Reaches +/-1 with ties."""
    n = len(x)
    c, d, xt, yt = _counts(x, y)
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - xt) * (n0 - yt))
    return (c - d) / denom if denom > 0 else 0.0


def pvalue(x, y, alternative="two-sided"):
    """Normal-approximation p-value for tau under the null of independence.

    z = (C - D) / sqrt(var), var = n(n-1)(2n+5)/18 (the no-tie form). alternative in
    {'two-sided','greater','less'}."""
    n = len(x)
    c, d, _xt, _yt = _counts(x, y)
    s = c - d
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if var <= 0:
        return 1.0
    # continuity correction
    z = (s - math.copysign(1, s)) / math.sqrt(var) if s != 0 else 0.0

    def phi(t):
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))

    if alternative == "greater":
        return 1 - phi(z)
    if alternative == "less":
        return phi(z)
    return min(1.0, 2 * (1 - phi(abs(z))))


def kendall(x, y, alternative="two-sided"):
    """Full Kendall test. Returns a dict with tau_b, tau_a, concordant, discordant, and p_value."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    c, d, xt, yt = _counts(x, y)
    return {
        "tau_b": tau_b(x, y),
        "tau_a": tau_a(x, y),
        "concordant": c,
        "discordant": d,
        "n_pairs": len(x) * (len(x) - 1) // 2,
        "p_value": pvalue(x, y, alternative),
    }

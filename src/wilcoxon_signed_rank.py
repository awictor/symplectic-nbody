"""Wilcoxon signed-rank test: the robust, distribution-free replacement for the paired t-test.

To ask whether a treatment shifted paired measurements -- before vs after, drug vs placebo on the same
subjects -- the paired t-test averages the differences and assumes they are normal. The WILCOXON
SIGNED-RANK TEST (1945) makes the weaker, robust assumption that the differences are merely SYMMETRIC about
their median, and uses their RANKS instead of their values. It ranks the ABSOLUTE differences, then sums
the ranks that came from POSITIVE differences:

    W+ = sum of ranks of the positive differences,   W- = sum of ranks of the negative,   W = min(W+, W-).

Zero differences are dropped; tied absolute differences get average ranks. Under the null of no shift, W+
has a known symmetric distribution -- computed EXACTLY for small samples by enumerating the 2^n sign
patterns, and approximated for larger n by a normal with mean n(n+1)/4 and variance n(n+1)(2n+1)/24 (with a
tie correction). Because it uses ranks, a single wild outlier moves W by at most one rank, where it would
swing the t-test's mean arbitrarily.

This module computes the signed-rank statistic (with average ranks for ties and zero-dropping), an exact
p-value by sign-pattern enumeration for small samples, a tie-corrected normal-approximation p-value for
larger ones, and the matched-pairs convenience wrapper. It is validated: a consistent positive shift gives
a tiny p-value and a symmetric-noise sample around zero gives a large one; the exact and normal p-values
agree for moderate n; the test is invariant to the scale of the differences (ranks only) and robust to an
outlier that would flip a t-test; W+ + W- equals n(n+1)/2; zeros are correctly dropped and ties averaged;
the exact null distribution is symmetric and sums to 2^n; it agrees with the repo's sign-flip permutation
test; and results are deterministic. Pure stdlib; the paired nonparametric-testing companion to the
Mann-Whitney, permutation-test, and sign-test tools."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mann_whitney import _average_ranks


def _signed_ranks(differences, zero_method="wilcox"):
    """Return (w_plus, w_minus, n_nonzero, ranks_of_abs). Drops zeros (wilcox convention)."""
    diffs = [d for d in differences if d != 0.0]
    if not diffs:
        return 0.0, 0.0, 0, []
    abs_d = [abs(d) for d in diffs]
    ranks = _average_ranks(abs_d)
    w_plus = sum(ranks[i] for i in range(len(diffs)) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(len(diffs)) if diffs[i] < 0)
    return w_plus, w_minus, len(diffs), ranks


def _exact_null_counts(n):
    """Distribution of W+ under the null: over all 2^n sign assignments of ranks 1..n, count how many
    give each value of sum(positive ranks). Returns a list counts[w] for w = 0..n(n+1)/2."""
    max_w = n * (n + 1) // 2
    counts = [0] * (max_w + 1)
    counts[0] = 1
    # each rank r (1..n) is either + (adds r) or - (adds 0): convolution
    for r in range(1, n + 1):
        new = [0] * (max_w + 1)
        for w in range(max_w + 1):
            if counts[w]:
                new[w] += counts[w]          # rank r negative
                if w + r <= max_w:
                    new[w + r] += counts[w]  # rank r positive
        counts = new
    return counts


def exact_pvalue(differences, alternative="two-sided"):
    """Exact p-value by enumerating the null distribution of W+ (small samples, assumes no ties in |d|)."""
    w_plus, w_minus, n, _ranks = _signed_ranks(differences)
    if n == 0:
        return 1.0
    counts = _exact_null_counts(n)
    total = sum(counts)
    max_w = n * (n + 1) // 2

    def p_le(x):
        x = int(math.floor(x + 1e-9))
        return sum(counts[:x + 1]) / total

    def p_ge(x):
        x = int(math.ceil(x - 1e-9))
        return sum(counts[x:]) / total

    if alternative == "greater":     # positive shift -> large W+
        return p_ge(w_plus)
    if alternative == "less":
        return p_le(w_plus)
    w = min(w_plus, w_minus)
    return min(1.0, 2.0 * p_le(w))


def normal_pvalue(differences, alternative="two-sided", continuity=True):
    """Tie-corrected normal-approximation p-value. Good for larger samples or tied |differences|."""
    w_plus, w_minus, n, ranks = _signed_ranks(differences)
    if n == 0:
        return 1.0
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    # tie correction: subtract sum(t^3 - t)/48 over tie groups of |d|
    from collections import Counter
    abs_d = [abs(d) for d in differences if d != 0.0]
    tie_term = sum(t ** 3 - t for t in Counter(abs_d).values())
    var -= tie_term / 48.0
    if var <= 0:
        return 1.0
    sigma = math.sqrt(var)
    z = w_plus - mu
    if continuity:
        z -= math.copysign(0.5, z)
    z /= sigma

    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    if alternative == "greater":
        return 1 - phi(z)
    if alternative == "less":
        return phi(z)
    return min(1.0, 2 * (1 - phi(abs(z))))


def wilcoxon(differences=None, x=None, y=None, alternative="two-sided"):
    """Wilcoxon signed-rank test. Provide either `differences` or paired samples x, y (uses x - y).

    Returns a dict with W (min of W+/W-), W_plus, W_minus, n (nonzero pairs), and p_value (exact for
    small untied samples, else tie-corrected normal approximation)."""
    if differences is None:
        if x is None or y is None:
            raise ValueError("provide differences or both x and y")
        differences = [x[i] - y[i] for i in range(len(x))]
    w_plus, w_minus, n, _ranks = _signed_ranks(differences)
    abs_d = [abs(d) for d in differences if d != 0.0]
    has_ties = len(set(abs_d)) < len(abs_d)
    use_exact = (n <= 20 and not has_ties)
    if use_exact:
        p = exact_pvalue(differences, alternative)
        method = "exact"
    else:
        p = normal_pvalue(differences, alternative)
        method = "normal-approx"
    return {"W": min(w_plus, w_minus), "W_plus": w_plus, "W_minus": w_minus,
            "n": n, "p_value": p, "method": method}

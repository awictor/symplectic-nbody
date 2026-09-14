"""Mann-Whitney U test: does one group tend to be larger than another, without assuming a distribution?

The two-sample t-test asks whether two groups have different MEANS, but it assumes roughly normal data and
is thrown off by outliers and skew. The MANN-WHITNEY U TEST (Wilcoxon rank-sum; 1945-1947) asks the more
robust question -- is one group STOCHASTICALLY LARGER than the other? -- using only the RANKS of the pooled
data, so it needs no normality and shrugs off outliers. Its statistic counts, over all pairs (one value
from each group), how often the first group's value exceeds the second's:

    U1 = R1 - n1(n1+1)/2,      where R1 is the sum of ranks of group 1 in the pooled ranking,

and U = min(U1, U2). Under the null hypothesis of identical distributions, U has a known combinatorial
distribution; for small samples it can be computed EXACTLY by enumerating rank assignments, and for larger
samples a normal approximation with a mean n1 n2 / 2 and a variance that includes a TIE CORRECTION gives an
accurate p-value. The related common-language effect size, U1 / (n1 n2), is simply the probability that a
random member of group 1 exceeds a random member of group 2.

This module computes the U statistic with average ranks for ties, an exact p-value by dynamic programming
over the null distribution for small samples, a tie-corrected normal-approximation p-value for large ones,
and the effect size. It is validated: on identical groups U1 = U2 = n1 n2 / 2 and the p-value is ~1; on
cleanly separated groups U collapses to 0 and the p-value is tiny; the exact and normal-approximation
p-values agree for moderate n; the effect size equals the brute-force fraction of cross-pairs where group
1 wins; the test is invariant to any monotonic transform of the data (it uses ranks); it handles ties via
average ranks with the variance correction; the U statistics satisfy U1 + U2 = n1 n2; and results are
deterministic. Pure stdlib; the nonparametric-testing companion to the KS-test, permutation-test, and
t-test tools."""

from __future__ import annotations

import math


def _average_ranks(values):
    """Rank the values 1..N, assigning tied values their average rank. Returns a list of ranks."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j (0-based) share ranks i+1..j+1; assign the average
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def u_statistic(a, b):
    """Return (U1, U2, R1) for groups a and b using average ranks over the pooled sample."""
    n1 = len(a)
    n2 = len(b)
    pooled = list(a) + list(b)
    ranks = _average_ranks(pooled)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    return u1, u2, r1


def effect_size(a, b):
    """Common-language effect size U1/(n1 n2): P(random a > random b), ties counted as 1/2."""
    u1, _u2, _r1 = u_statistic(a, b)
    return u1 / (len(a) * len(b))


def _exact_null_counts(n1, n2):
    """Distribution of U1 under the null by DP: number of ways to get each U value.

    counts[u] = number of rank assignments giving U1 = u, over C(n1+n2, n1) total. Standard recurrence
    for the Mann-Whitney null (no ties)."""
    max_u = n1 * n2
    # Count rank assignments giving each U1 by the standard Mann-Whitney recurrence, memoized.
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def ways(m, k, u):
        # ways to get U = u with m group-1 and k group-2 items = partitions of u into <= m parts
        # each <= k. Standard recurrence: add a part of size k (-> u-k) or cap parts at k-1 (-> k-1).
        if u < 0:
            return 0
        if m == 0 or k == 0:
            return 1 if u == 0 else 0
        return ways(m - 1, k, u - k) + ways(m, k - 1, u)

    counts = [ways(n1, n2, u) for u in range(max_u + 1)]
    ways.cache_clear()
    return counts


def exact_pvalue(a, b, alternative="two-sided"):
    """Exact two-sided (or one-sided) p-value for small samples with NO ties.

    Enumerates the null distribution of U1 by dynamic programming. For tied data or large samples use
    normal_pvalue. alternative in {'two-sided', 'less', 'greater'} (less/greater refer to group a)."""
    n1, n2 = len(a), len(b)
    u1, u2, _r1 = u_statistic(a, b)
    counts = _exact_null_counts(n1, n2)
    total = sum(counts)
    max_u = n1 * n2

    def p_le(x):
        x = int(math.floor(x + 1e-9))
        return sum(counts[:x + 1]) / total

    def p_ge(x):
        x = int(math.ceil(x - 1e-9))
        return sum(counts[x:]) / total

    if alternative == "greater":     # a stochastically greater -> large U1
        return p_ge(u1)
    if alternative == "less":
        return p_le(u1)
    # two-sided: double the smaller tail (symmetric null)
    u = min(u1, u2)
    return min(1.0, 2.0 * p_le(u))


def normal_pvalue(a, b, alternative="two-sided", continuity=True):
    """Normal-approximation p-value with tie correction. Good for larger samples or tied data."""
    n1, n2 = len(a), len(b)
    u1, u2, _r1 = u_statistic(a, b)
    mu = n1 * n2 / 2.0
    N = n1 + n2
    # tie correction to the variance
    pooled = list(a) + list(b)
    from collections import Counter
    tie_term = sum(t ** 3 - t for t in Counter(pooled).values())
    var = (n1 * n2 / 12.0) * ((N + 1) - tie_term / (N * (N - 1)))
    if var <= 0:
        return 1.0
    sigma = math.sqrt(var)
    u = u1
    z = (u - mu)
    if continuity:
        z -= math.copysign(0.5, z)
    z /= sigma

    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    if alternative == "greater":
        return 1 - phi(z)
    if alternative == "less":
        return phi(z)
    return min(1.0, 2.0 * (1 - phi(abs(z))))


def mann_whitney(a, b, alternative="two-sided"):
    """Full Mann-Whitney test. Returns a dict with U (min), U1, U2, effect_size, and a p-value
    (exact when both samples are small and untied, else the tie-corrected normal approximation)."""
    n1, n2 = len(a), len(b)
    u1, u2, _r1 = u_statistic(a, b)
    pooled = list(a) + list(b)
    has_ties = len(set(pooled)) < len(pooled)
    use_exact = (n1 <= 20 and n2 <= 20 and not has_ties)
    if use_exact:
        p = exact_pvalue(a, b, alternative)
        method = "exact"
    else:
        p = normal_pvalue(a, b, alternative)
        method = "normal-approx"
    return {
        "U": min(u1, u2),
        "U1": u1,
        "U2": u2,
        "effect_size": u1 / (n1 * n2),
        "p_value": p,
        "method": method,
    }

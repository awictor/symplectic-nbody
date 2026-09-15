"""Friedman test: nonparametric repeated-measures ANOVA -- do k treatments differ across matched blocks?

When the same subjects (or matched blocks) are measured under several conditions -- three drugs on each
patient, five algorithms on each dataset, four raters scoring each item -- a repeated-measures ANOVA tests
whether the conditions differ, but assumes normal, equal-variance data. The FRIEDMAN TEST (1937) is its
nonparametric replacement, and the blocked analogue of Kruskal-Wallis. Within EACH block it ranks the k
treatments 1..k, then asks whether the treatments' average ranks differ more than chance:

    Q = 12 / (n k (k+1)) * sum_j (R_j - n(k+1)/2)^2,

where n is the number of blocks, k the treatments, and R_j the rank sum of treatment j. Ranking WITHIN
each block cancels block-to-block level differences (a patient who scores high on everything), isolating
the treatment effect. Under the null of no treatment difference Q is approximately chi-squared with k-1
degrees of freedom; ties within a block get average ranks and a divisor correction. When a significant Q
is found, the average ranks say which treatments lead.

This module computes the Friedman Q statistic with tie correction, its chi-squared p-value (reusing the
repo's chi-squared survival function), the average rank of each treatment, and Kendall's W coefficient of
concordance (Q normalized to [0,1], measuring agreement among blocks). It is validated: identical
treatments give small Q and a large p, one clearly-superior treatment gives large Q and a tiny p, the test
removes additive per-block effects (adding a constant to a whole block changes nothing), for k=2 it reduces
to a sign-test-style comparison, ranking within blocks is what distinguishes it from Kruskal-Wallis, ties
are handled by average ranks, Kendall's W lies in [0,1] and hits 1 for perfect agreement, and results are
deterministic. Pure stdlib; the repeated-measures nonparametric companion to the Kruskal-Wallis,
Wilcoxon-signed-rank, and Mann-Whitney tools."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mann_whitney import _average_ranks
from kruskal_wallis import chi2_sf


def _rank_within_blocks(data):
    """data is a list of blocks, each a list of k treatment values. Returns the ranks per block."""
    return [_average_ranks(block) for block in data]


def friedman(data):
    """Friedman test. data: list of n blocks, each a list of k treatment measurements (same order).

    Returns a dict with Q, df, p_value, rank_sums, avg_ranks, and kendall_w. Ranks treatments within
    each block, then tests whether the treatments' rank sums differ."""
    n = len(data)
    if n < 2:
        raise ValueError("need at least 2 blocks")
    k = len(data[0])
    if k < 2:
        raise ValueError("need at least 2 treatments")
    if any(len(b) != k for b in data):
        raise ValueError("all blocks must have the same number of treatments")

    ranks = _rank_within_blocks(data)
    rank_sums = [sum(ranks[i][j] for i in range(n)) for j in range(k)]
    expected = n * (k + 1) / 2.0
    ss = sum((R - expected) ** 2 for R in rank_sums)

    # tie correction: divide by 1 - sum(t^3 - t) / (n(k^3 - k))
    from collections import Counter
    tie_sum = 0
    for block in data:
        for t in Counter(block).values():
            tie_sum += t ** 3 - t
    denom_correction = 1.0 - tie_sum / (n * (k ** 3 - k)) if (k ** 3 - k) > 0 else 1.0

    q = 12.0 / (n * k * (k + 1)) * ss
    if denom_correction > 0:
        q /= denom_correction

    df = k - 1
    p = chi2_sf(q, df)
    avg_ranks = [R / n for R in rank_sums]
    # Kendall's W = Q / (n(k-1)), coefficient of concordance in [0,1]
    kendall_w = q / (n * (k - 1)) if n * (k - 1) > 0 else 0.0
    return {
        "Q": q,
        "df": df,
        "p_value": p,
        "rank_sums": rank_sums,
        "avg_ranks": avg_ranks,
        "kendall_w": min(1.0, kendall_w),
    }


def kendall_w(data):
    """Kendall's coefficient of concordance W in [0,1]: agreement among the blocks' rankings."""
    return friedman(data)["kendall_w"]

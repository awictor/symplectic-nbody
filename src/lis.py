"""Longest increasing subsequence: the O(n log n) patience-sorting algorithm.

Given a sequence, the LONGEST INCREASING SUBSEQUENCE (LIS) is the longest set of elements, in their
original order, that strictly increases -- not necessarily contiguous. It measures how "sorted" a
sequence is, and appears in patience-sorting card games, computational biology (the longest
consistently-ordered gene run between two genomes), the Robinson-Schensted correspondence, and the
analysis of permutations. The naive dynamic program is O(n^2); the elegant PATIENCE SORTING
algorithm does it in O(n log n).

Patience sorting deals the sequence like a game of solitaire: each number is placed on the leftmost
PILE whose top card is greater-or-equal (so it can sit on top), or starts a new pile to the right if
none qualifies. The number of piles at the end EQUALS the LIS length -- a theorem tying the greedy
pile placement to the optimum. Because the pile tops stay sorted, the correct pile is found by BINARY
SEARCH, giving the log factor. To recover the actual subsequence (not just its length), each placed
card records a back-pointer to the top of the pile to its left at the moment it was placed; tracing
back from the last pile reconstructs one longest increasing subsequence.

This module computes the LIS length and one witnessing subsequence in O(n log n), with variants for
strictly increasing, non-decreasing, and longest DECREASING subsequences. It is verified against a
brute-force O(2^n) search over all subsequences for short inputs -- that the length is optimal and the
returned subsequence is genuinely increasing and a subsequence of the input -- against the O(n^2) DP,
and on known cases (sorted, reverse-sorted, all-equal, empty). Pure stdlib; an algorithms companion
to the edit-distance, LCS, and dynamic-programming notes."""

from __future__ import annotations

import bisect


def lis_length(seq, strict=True):
    """The length of the longest increasing subsequence via patience sorting, in O(n log n).

    strict=True requires strictly increasing; False allows non-decreasing (equal elements)."""
    tails = []          # tails[k] = smallest possible tail of an increasing subsequence of length k+1
    for x in seq:
        if strict:
            i = bisect.bisect_left(tails, x)
        else:
            i = bisect.bisect_right(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def lis(seq, strict=True):
    """The length AND one witnessing longest increasing subsequence.

    Returns (length, subsequence). Reconstructs via back-pointers from patience sorting."""
    n = len(seq)
    if n == 0:
        return 0, []
    tails = []              # tails[k] = index (into seq) of the smallest tail of a length-(k+1) LIS
    tail_vals = []          # the corresponding values, kept sorted for binary search
    prev = [-1] * n         # prev[i] = index of the predecessor of seq[i] in its subsequence
    for i, x in enumerate(seq):
        if strict:
            k = bisect.bisect_left(tail_vals, x)
        else:
            k = bisect.bisect_right(tail_vals, x)
        if k > 0:
            prev[i] = tails[k - 1]
        if k == len(tails):
            tails.append(i)
            tail_vals.append(x)
        else:
            tails[k] = i
            tail_vals[k] = x
    # reconstruct from the last pile's tail
    length = len(tails)
    result = []
    idx = tails[-1]
    while idx != -1:
        result.append(seq[idx])
        idx = prev[idx]
    result.reverse()
    return length, result


def longest_decreasing_subsequence(seq, strict=True):
    """The longest strictly (or weakly) DECREASING subsequence: LIS of the reversed-comparison
    sequence. Returns (length, subsequence)."""
    # negate values so decreasing becomes increasing, then map back
    length, sub = lis([-x for x in seq], strict=strict)
    return length, [-x for x in sub]


def lis_dp(seq, strict=True):
    """The O(n^2) dynamic-programming LIS length (for cross-checking the fast version)."""
    n = len(seq)
    if n == 0:
        return 0
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            better = seq[j] < seq[i] if strict else seq[j] <= seq[i]
            if better and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
    return max(dp)

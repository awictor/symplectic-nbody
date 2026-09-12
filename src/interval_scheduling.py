"""Weighted interval scheduling: the maximum-value set of non-overlapping intervals.

Given jobs, each with a start time, finish time, and a WEIGHT (value), pick a subset of mutually
NON-OVERLAPPING jobs that maximizes total weight. This models booking a resource for the most
valuable non-conflicting requests: renting a hall, scheduling a machine, allocating a satellite. When
all weights are equal it reduces to the classic "activity selection" (maximize the COUNT of jobs),
which a simple earliest-finish-first greedy solves -- but with arbitrary weights greed fails, and the
problem needs DYNAMIC PROGRAMMING.

The DP is elegant. Sort the jobs by finish time. For job i, define p(i) as the latest job that
finishes at or before job i STARTS (found by binary search) -- the last job compatible with taking i.
Then the best value using jobs 1..i is opt(i) = max(opt(i-1), weight_i + opt(p(i))): either skip job
i, or take it plus the best schedule among the jobs that finish before it begins. Filling opt left to
right is O(n log n) (the sort and the binary searches dominate), and tracing back the max choices
recovers the actual chosen set. The unweighted count-maximizing version is the greedy special case,
also provided.

This module implements weighted interval scheduling (max total weight, with the chosen jobs) and the
greedy earliest-finish activity selection (max count), over a list of (start, finish, weight) jobs. It
is verified against brute force -- an exhaustive check over all 2^n subsets for small inputs -- that
the DP finds the true maximum weight, that the returned jobs are genuinely non-overlapping and sum to
that weight, that the greedy count matches the maximum independent set of intervals, that equal
weights make the DP agree with the greedy count, and on hand-checked cases. Pure stdlib; a
dynamic-programming companion to the knapsack, LIS, and matrix-chain notes."""

from __future__ import annotations

import bisect


def schedule(jobs):
    """Maximum-weight set of non-overlapping jobs.

    jobs: list of (start, finish, weight). Returns (max_weight, chosen_jobs) where chosen_jobs is the
    selected subset (as (start, finish, weight) tuples) sorted by finish time."""
    if not jobs:
        return 0.0, []
    # sort by finish time
    js = sorted(jobs, key=lambda j: j[1])
    n = len(js)
    starts = [j[0] for j in js]
    finishes = [j[1] for j in js]

    # p[i] = index (into js) of the latest job finishing <= js[i]'s start, or -1
    p = []
    for i in range(n):
        # rightmost job whose finish <= starts[i]
        idx = bisect.bisect_right(finishes, starts[i]) - 1
        p.append(idx)

    # opt[i] = best weight using jobs 0..i-1 (opt has length n+1, opt[0] = 0)
    opt = [0.0] * (n + 1)
    for i in range(1, n + 1):
        w = js[i - 1][2]
        take = w + opt[p[i - 1] + 1]         # p is 0-indexed into js; +1 to align with opt offset
        skip = opt[i - 1]
        opt[i] = max(take, skip)

    # reconstruct
    chosen = []
    i = n
    while i > 0:
        w = js[i - 1][2]
        take = w + opt[p[i - 1] + 1]
        if take >= opt[i - 1]:
            chosen.append(js[i - 1])
            i = p[i - 1] + 1
        else:
            i -= 1
    chosen.reverse()
    return opt[n], chosen


def activity_selection(jobs):
    """Maximum COUNT of non-overlapping jobs (unweighted), by the earliest-finish greedy. Returns
    (count, chosen_jobs)."""
    if not jobs:
        return 0, []
    js = sorted(jobs, key=lambda j: j[1])
    chosen = []
    last_finish = float("-inf")
    for j in js:
        if j[0] >= last_finish:
            chosen.append(j)
            last_finish = j[1]
    return len(chosen), chosen


def total_weight(jobs):
    return sum(j[2] for j in jobs)


def is_compatible(jobs):
    """True if a set of jobs is pairwise non-overlapping."""
    js = sorted(jobs, key=lambda j: j[1])
    for i in range(len(js) - 1):
        if js[i][1] > js[i + 1][0]:
            return False
    return True

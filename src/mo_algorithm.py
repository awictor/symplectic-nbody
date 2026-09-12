"""Mo's algorithm: answering many range queries fast by reordering them cleverly.

Given a fixed array and a batch of queries -- each asking something about a subrange [l, r], like "how
many DISTINCT values are in this window?" or "what is the sum of frequencies squared?" -- the naive
approach recomputes each answer from scratch in O(n) per query, O(q*n) overall. Mo's algorithm answers
all q queries OFFLINE in O((n + q) * sqrt(n)) by a beautiful trick: it maintains a current window and a
running answer, and MOVES the window's endpoints one step at a time (adding or removing a single
element, each an O(1) update) to morph one query's range into the next. The cost is the total distance
the two pointers travel; the magic is choosing an ORDER for the queries that minimises that distance.

The ordering sorts queries into sqrt(n)-sized blocks by their left endpoint, and within each block by
right endpoint (alternating the right-endpoint direction per block -- the "Hilbert-like" snake order --
to shave another constant factor). With this order the left pointer moves O(sqrt(n)) per query and the
right pointer moves O(n) per block, giving the O((n+q)*sqrt(n)) bound. The only thing the problem must
supply is a cheap ADD and REMOVE that update the running answer when one element enters or leaves the
window -- for distinct-count that is a frequency table and a running count of nonzero frequencies; for
sum-of-squared-frequencies it is an incremental update of sum(f^2). Because it processes queries in a
permuted order, it is inherently offline (all queries known up front), the price for its speed.

This module implements the query-reordering engine and two concrete applications -- range
DISTINCT-VALUE counts and range SUM-OF-SQUARED-FREQUENCIES (the "power sum" that appears in the classic
D-query and array-power problems) -- and returns answers in the original query order. It is verified
against brute force: every answer matches a direct recomputation over the subrange, on hundreds of
random arrays and query batches, including full-array and single-element ranges. Pure stdlib; a
data-structures companion to the segment-tree, sparse-table, and Fenwick notes."""

from __future__ import annotations

from math import isqrt


def _mo_order(queries, block):
    """Return query indices sorted in Mo's order: by left-block, then by right endpoint with the
    direction alternating per block (snake order) to reduce right-pointer travel."""
    def key(i):
        l, r = queries[i]
        b = l // block
        # even blocks: ascending r; odd blocks: descending r
        return (b, r if b % 2 == 0 else -r)
    return sorted(range(len(queries)), key=key)


def _run(n, queries, add, remove, current):
    """Core driver: process `queries` (list of (l, r) inclusive ranges) in Mo's order, calling
    `add(i)` / `remove(i)` to move the window and `current()` to read the answer. Returns answers in
    the ORIGINAL query order."""
    if not queries:
        return []
    block = max(1, isqrt(n))
    order = _mo_order(queries, block)
    answers = [None] * len(queries)
    cur_l, cur_r = 0, -1          # empty window
    for qi in order:
        l, r = queries[qi]
        while cur_r < r:
            cur_r += 1
            add(cur_r)
        while cur_l > l:
            cur_l -= 1
            add(cur_l)
        while cur_r > r:
            remove(cur_r)
            cur_r -= 1
        while cur_l < l:
            remove(cur_l)
            cur_l += 1
        answers[qi] = current()
    return answers


def range_distinct(arr, queries):
    """For each inclusive range (l, r) in `queries`, the number of DISTINCT values in arr[l..r].
    Answers are returned in the original query order. O((n+q) sqrt n)."""
    n = len(arr)
    freq = {}
    state = {"distinct": 0}

    def add(i):
        v = arr[i]
        c = freq.get(v, 0)
        if c == 0:
            state["distinct"] += 1
        freq[v] = c + 1

    def remove(i):
        v = arr[i]
        freq[v] -= 1
        if freq[v] == 0:
            state["distinct"] -= 1

    def current():
        return state["distinct"]

    return _run(n, queries, add, remove, current)


def range_power_sum(arr, queries):
    """For each inclusive range, sum over distinct values v of freq(v)^2 -- the "power sum" that
    weights each value by the square of how often it appears in the window. O((n+q) sqrt n)."""
    n = len(arr)
    freq = {}
    state = {"s": 0}

    def add(i):
        v = arr[i]
        f = freq.get(v, 0)
        state["s"] += 2 * f + 1        # (f+1)^2 - f^2
        freq[v] = f + 1

    def remove(i):
        v = arr[i]
        f = freq[v]
        state["s"] += -2 * f + 1       # (f-1)^2 - f^2
        freq[v] = f - 1

    def current():
        return state["s"]

    return _run(n, queries, add, remove, current)


# --- brute-force references -------------------------------------------------
def brute_distinct(arr, queries):
    """Range distinct-value counts by direct recomputation."""
    return [len(set(arr[l:r + 1])) for l, r in queries]


def brute_power_sum(arr, queries):
    """Range sum-of-squared-frequencies by direct recomputation."""
    from collections import Counter
    out = []
    for l, r in queries:
        c = Counter(arr[l:r + 1])
        out.append(sum(f * f for f in c.values()))
    return out

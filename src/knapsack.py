"""The 0/1 knapsack: packing the most value under a weight limit.

A thief with a bag that holds W kilograms faces items each with a weight and a value; which
subset maximizes total value without exceeding W? Each item is taken whole or left behind (0 or
1 -- no fractions), and that "all-or-nothing" constraint makes the problem NP-hard by brute force
(2^n subsets) yet solvable in pseudo-polynomial O(nW) time by dynamic programming. It is the
textbook model for budget allocation, cargo loading, cutting stock, and portfolio selection under
a hard cap.

The recurrence: let best[i][w] be the most value achievable from the first i items within
capacity w. Item i is either skipped or taken:

    best[i][w] = max( best[i-1][w],                              skip item i
                      best[i-1][w - weight_i] + value_i )        take it (if it fits).

Filling the table row by row gives the optimum; following the choices backward recovers WHICH
items to take. A one-dimensional rolling array (iterating capacity downward so each item is used
at most once) does it in O(W) memory. The related SUBSET-SUM question -- can any subset hit
exactly a target weight? -- is the same table with a boolean reachability recurrence, and the
UNBOUNDED knapsack (unlimited copies) just iterates capacity upward instead.

This module solves the 0/1 knapsack for the optimal value and the chosen items, the space-lean
value-only version, subset-sum, and unbounded knapsack, and checks them against a brute-force
search over all subsets. Pure stdlib; the dynamic-programming companion to the Levenshtein note.
"""

from __future__ import annotations


def knapsack(weights, values, capacity: int):
    """Maximum value for the 0/1 knapsack and the indices of the chosen items. Returns
    (best_value, chosen_indices). Weights and capacity are nonnegative integers."""
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")
    n = len(weights)
    best = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi, vi = weights[i - 1], values[i - 1]
        for w in range(capacity + 1):
            best[i][w] = best[i - 1][w]                       # skip
            if wi <= w:
                take = best[i - 1][w - wi] + vi               # take
                if take > best[i][w]:
                    best[i][w] = take
    # backtrace the chosen items
    chosen = []
    w = capacity
    for i in range(n, 0, -1):
        if best[i][w] != best[i - 1][w]:                      # item i was taken
            chosen.append(i - 1)
            w -= weights[i - 1]
    chosen.reverse()
    return best[n][capacity], chosen


def knapsack_value(weights, values, capacity: int) -> int:
    """Optimal 0/1 knapsack value using a single rolling array -- O(capacity) memory. Iterating
    capacity downward ensures each item is used at most once."""
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")
    dp = [0] * (capacity + 1)
    for wi, vi in zip(weights, values):
        for w in range(capacity, wi - 1, -1):
            cand = dp[w - wi] + vi
            if cand > dp[w]:
                dp[w] = cand
    return dp[capacity]


def subset_sum(weights, target: int):
    """Can a subset of `weights` sum to exactly `target`? Returns the subset's indices if so,
    else None. Uses the reachability DP."""
    if target < 0:
        return None
    n = len(weights)
    reach = [[False] * (target + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        reach[i][0] = True                                    # empty subset hits 0
    for i in range(1, n + 1):
        wi = weights[i - 1]
        for t in range(target + 1):
            reach[i][t] = reach[i - 1][t]
            if wi <= t and reach[i - 1][t - wi]:
                reach[i][t] = True
    if not reach[n][target]:
        return None
    chosen = []
    t = target
    for i in range(n, 0, -1):
        if not reach[i - 1][t] and t >= weights[i - 1] and reach[i - 1][t - weights[i - 1]]:
            chosen.append(i - 1)
            t -= weights[i - 1]
    chosen.reverse()
    return chosen


def unbounded_knapsack(weights, values, capacity: int) -> int:
    """Maximum value when each item may be taken any number of times (unbounded knapsack).
    Iterating capacity UPWARD lets an item be reused."""
    if capacity < 0:
        raise ValueError("capacity must be nonnegative")
    dp = [0] * (capacity + 1)
    for w in range(1, capacity + 1):
        for wi, vi in zip(weights, values):
            if wi <= w:
                cand = dp[w - wi] + vi
                if cand > dp[w]:
                    dp[w] = cand
    return dp[capacity]


# --- brute-force reference (for validation) --------------------------------

def brute_knapsack(weights, values, capacity: int):
    """Best value over all 2^n subsets, for checking the DP. Returns (value, indices)."""
    n = len(weights)
    best_val, best_set = 0, []
    for mask in range(1 << n):
        w = v = 0
        items = []
        for i in range(n):
            if mask & (1 << i):
                w += weights[i]
                v += values[i]
                items.append(i)
        if w <= capacity and v > best_val:
            best_val, best_set = v, items
    return best_val, best_set

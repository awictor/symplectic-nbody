"""Coin change: fewest coins to make an amount, and how many ways to make it.

Given coin denominations and a target amount, two classic questions: the MINIMUM number of coins that
sum to the amount, and the NUMBER OF WAYS to make it. Both are textbook DYNAMIC PROGRAMMING, and both
expose a trap: the obvious GREEDY (always take the largest coin that fits) gives the right minimum for
"canonical" currency systems like US coins, but FAILS for many denomination sets -- e.g. with coins
{1, 3, 4} making 6, greedy takes 4+1+1 = 3 coins but the optimum is 3+3 = 2. Only DP is guaranteed
correct, which is why change-making, cashier systems, and stamp/postage problems all rest on it.

The MINIMUM-COINS DP builds up min_coins[a] = 1 + min over coins c <= a of min_coins[a - c], with
min_coins[0] = 0 and infinity where no combination works; recording which coin achieved each minimum
reconstructs the actual coin multiset. The COUNTING DP is subtler: to count COMBINATIONS (order-
independent) rather than sequences, iterate the coins in the OUTER loop and the amount in the inner
loop, so ways[a] += ways[a - c] accumulates each coin's contribution once -- swapping the loop order
would instead count ordered sequences (compositions). Both run in O(amount * #coins).

This module computes the minimum number of coins (with the coins used), whether an amount is makeable,
and the number of distinct combinations, for arbitrary denominations. It is verified against brute
force -- an exhaustive search over coin multisets for small amounts -- that the minimum is truly
minimal and the reconstructed coins sum to the amount, that unmakeable amounts are detected, that the
combination count matches a brute enumeration, that a greedy-defeating denomination set is handled
correctly, and on canonical currency where greedy and DP agree. Pure stdlib; a dynamic-programming
companion to the knapsack, interval-scheduling, and LIS notes."""

from __future__ import annotations


def min_coins(coins, amount):
    """The fewest coins summing to `amount`, and the coin multiset. Returns (count, coins_used), or
    (None, None) if the amount cannot be made."""
    if amount < 0:
        return None, None
    INF = float("inf")
    best = [0] + [INF] * amount
    choice = [-1] * (amount + 1)             # which coin was used to reach each amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and best[a - c] + 1 < best[a]:
                best[a] = best[a - c] + 1
                choice[a] = c
    if best[amount] == INF:
        return None, None
    # reconstruct
    used = []
    a = amount
    while a > 0:
        c = choice[a]
        used.append(c)
        a -= c
    used.sort()
    return best[amount], used


def can_make(coins, amount):
    """True if `amount` can be formed from the coins."""
    return min_coins(coins, amount)[0] is not None


def count_ways(coins, amount):
    """The number of distinct COMBINATIONS (order-independent) of coins summing to amount."""
    if amount < 0:
        return 0
    ways = [1] + [0] * amount                # one way to make 0: use nothing
    for c in coins:                          # coins in the OUTER loop -> combinations, not sequences
        for a in range(c, amount + 1):
            ways[a] += ways[a - c]
    return ways[amount]


def count_sequences(coins, amount):
    """The number of ordered SEQUENCES (compositions) of coins summing to amount (order matters)."""
    if amount < 0:
        return 0
    ways = [1] + [0] * amount
    for a in range(1, amount + 1):           # amount in the outer loop -> sequences
        for c in coins:
            if c <= a:
                ways[a] += ways[a - c]
    return ways[amount]

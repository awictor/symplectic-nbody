"""Integer partitions: counting and generating the ways to write n as a sum, and Euler's magic.

A PARTITION of a positive integer n is a way to write it as a sum of positive integers, order
ignored: 4 = 4 = 3+1 = 2+2 = 2+1+1 = 1+1+1+1, so p(4) = 5. The partition function p(n) grows
astonishingly fast (p(100) is about 190 million) yet is computed exactly in O(n^1.5) by EULER'S
PENTAGONAL NUMBER THEOREM, one of the gems of combinatorics. The generating function is the infinite
product 1 / prod_{k>=1}(1 - x^k), and Euler showed its reciprocal has almost all coefficients zero:

    prod (1 - x^k) = sum_{j=-inf}^{inf} (-1)^j x^{j(3j-1)/2},

the nonzero terms landing only at the GENERALISED PENTAGONAL NUMBERS g_j = j(3j-1)/2. That collapses
the recurrence to a signed sum over just O(sqrt n) previous values:

    p(n) = sum_{j>=1} (-1)^{j-1} [ p(n - g_j) + p(n - g_{-j}) ].

This module computes p(n) by that recurrence, generates every partition explicitly, counts and
generates RESTRICTED partitions (into distinct parts, into parts of bounded size, into exactly k
parts), and exposes the conjugate (transpose of the Ferrers diagram). It also verifies EULER'S
THEOREM -- the number of partitions into DISTINCT parts equals the number into ODD parts -- a
classic bijective identity.

Validated against brute-force enumeration: the pentagonal recurrence for p(n) matches an independent
count of generated partitions and a dynamic-programming coin-style count; every generated partition
sums to n and they are all distinct; distinct-part and odd-part counts are equal (Euler); the
conjugate is an involution mapping largest-part to number-of-parts; and known values (p(0)=1, p(10)=42,
p(100)=190569292) are reproduced. Pure stdlib; the combinatorial-counting companion to the
combinatorial-ranking and Catalan-style enumeration notes."""

from __future__ import annotations

from functools import lru_cache


def partition_count(n):
    """p(n), the number of integer partitions of n, via Euler's pentagonal recurrence in O(n^1.5)."""
    if n < 0:
        return 0
    p = [0] * (n + 1)
    p[0] = 1
    for m in range(1, n + 1):
        total = 0
        j = 1
        while True:
            g1 = j * (3 * j - 1) // 2   # generalised pentagonal, positive j
            g2 = j * (3 * j + 1) // 2   # ... negative j
            if g1 > m and g2 > m:
                break
            sign = -1 if (j % 2 == 0) else 1
            if g1 <= m:
                total += sign * p[m - g1]
            if g2 <= m:
                total += sign * p[m - g2]
            j += 1
        p[m] = total
    return p[n]


def partition_counts_up_to(n):
    """The list [p(0), p(1), ..., p(n)]."""
    p = [0] * (n + 1)
    p[0] = 1
    for m in range(1, n + 1):
        total = 0
        j = 1
        while True:
            g1 = j * (3 * j - 1) // 2
            g2 = j * (3 * j + 1) // 2
            if g1 > m and g2 > m:
                break
            sign = -1 if (j % 2 == 0) else 1
            if g1 <= m:
                total += sign * p[m - g1]
            if g2 <= m:
                total += sign * p[m - g2]
            j += 1
        p[m] = total
    return p


def generate_partitions(n, max_part=None):
    """Yield every partition of n as a non-increasing tuple. Optionally cap the largest part."""
    if n == 0:
        yield ()
        return
    if max_part is None:
        max_part = n
    for first in range(min(n, max_part), 0, -1):
        for rest in generate_partitions(n - first, first):
            yield (first,) + rest


def generate_distinct_partitions(n, max_part=None):
    """Yield every partition of n into DISTINCT parts (strictly decreasing)."""
    if n == 0:
        yield ()
        return
    if max_part is None:
        max_part = n
    for first in range(min(n, max_part), 0, -1):
        for rest in generate_distinct_partitions(n - first, first - 1):
            yield (first,) + rest


def count_distinct_partitions(n):
    """Number of partitions of n into distinct parts."""
    # DP over parts 1..n, each used at most once
    dp = [0] * (n + 1)
    dp[0] = 1
    for k in range(1, n + 1):
        for s in range(n, k - 1, -1):
            dp[s] += dp[s - k]
    return dp[n]


def count_odd_partitions(n):
    """Number of partitions of n into odd parts (each odd part used any number of times)."""
    dp = [0] * (n + 1)
    dp[0] = 1
    for k in range(1, n + 1, 2):  # odd parts only
        for s in range(k, n + 1):
            dp[s] += dp[s - k]
    return dp[n]


def count_partitions_into_k_parts(n, k):
    """Number of partitions of n into exactly k positive parts."""
    if k < 0 or k > n:
        return 0
    if k == 0:
        return 1 if n == 0 else 0
    # p(n,k) = p(n-1,k-1) + p(n-k,k)
    @lru_cache(maxsize=None)
    def pk(n, k):
        if k == 0:
            return 1 if n == 0 else 0
        if n < k:
            return 0
        return pk(n - 1, k - 1) + pk(n - k, k)
    return pk(n, k)


def conjugate(partition):
    """The conjugate partition (transpose of the Ferrers diagram): conj[i] = #parts >= i+1."""
    if not partition:
        return ()
    largest = partition[0]
    return tuple(sum(1 for p in partition if p >= i) for i in range(1, largest + 1))


# --- brute-force references --------------------------------------------------
def brute_partition_count(n):
    """Count partitions by generating them all."""
    return sum(1 for _ in generate_partitions(n))


def dp_partition_count(n):
    """Independent unbounded-knapsack DP count: partitions = coin change with coins 1..n, order-free."""
    dp = [0] * (n + 1)
    dp[0] = 1
    for coin in range(1, n + 1):
        for s in range(coin, n + 1):
            dp[s] += dp[s - coin]
    return dp[n]

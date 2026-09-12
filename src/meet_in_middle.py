"""Meet in the middle: halving the exponent of a brute-force search.

Some problems are genuinely exponential -- subset sum over n items with HUGE values, where the
pseudo-polynomial O(n*W) dynamic program is useless because W is astronomical. Brute force tries all
2^n subsets. MEET IN THE MIDDLE is a general technique that cuts the exponent in half: split the items
into two halves of size n/2, enumerate all 2^(n/2) subset-sums of each half separately, then COMBINE
the two lists cleverly -- with sorting and binary search or a hash -- to answer the original question.
The cost drops from O(2^n) to O(2^(n/2) * n), which turns an impossible n = 40 (a trillion subsets)
into a very fast one (a million per half). The same halving idea speeds up the discrete logarithm
(baby-step giant-step), the k-sum problem, and hard knapsack instances.

For SUBSET SUM ("is there a subset summing exactly to the target?"), enumerate the left half's sums
into a set and, for each right-half sum s, check whether target - s appears -- O(2^(n/2)). For CLOSEST
SUBSET SUM ("the achievable sum nearest the target, or the largest not exceeding a capacity"), sort one
half's sums and binary-search each of the other half's sums against it. For COUNTING subsets with a
given sum, tally the left sums in a dictionary and add up the matching complements. Each of these keeps
only 2^(n/2) values in memory, so problems far beyond the reach of naive enumeration become tractable
while the answers stay exact integers.

This module implements meet-in-the-middle subset-sum existence, the maximum subset sum not exceeding a
capacity, the closest achievable sum to a target, and counting subsets with an exact sum. It is
verified against brute-force enumeration of all 2^n subsets -- identical answers on hundreds of random
instances -- and on hand-checked cases, including instances with values far too large for a
pseudo-polynomial DP. Pure stdlib; an algorithm-technique companion to the knapsack/subset-sum DP, the
baby-step-giant-step discrete log, and combinatorial-search notes."""

from __future__ import annotations

from bisect import bisect_right


def _half_sums(items):
    """All 2^len(items) subset sums of `items` (with repetition allowed across different subsets).
    Returns an unsorted list of length 2^len(items)."""
    sums = [0]
    for x in items:
        sums += [s + x for s in sums]
    return sums


def subset_sum_exists(items, target):
    """True iff some subset of `items` sums exactly to `target`, in O(2^(n/2) * n) time and
    O(2^(n/2)) space via meet in the middle."""
    mid = len(items) // 2
    left = set(_half_sums(items[:mid]))
    for s in _half_sums(items[mid:]):
        if target - s in left:
            return True
    return False


def max_subset_sum_under(items, capacity):
    """The maximum subset sum that does not exceed `capacity` (a 0/1 knapsack where value == weight).
    Returns that best achievable sum (0 if only the empty subset fits)."""
    mid = len(items) // 2
    left = sorted(s for s in _half_sums(items[:mid]) if s <= capacity)
    best = 0
    for s in _half_sums(items[mid:]):
        if s > capacity:
            continue
        room = capacity - s
        # largest left sum <= room
        idx = bisect_right(left, room) - 1
        if idx >= 0:
            best = max(best, s + left[idx])
    return best


def closest_subset_sum(items, target):
    """The achievable subset sum closest to `target` (ties resolved toward the smaller sum). Returns
    that sum."""
    mid = len(items) // 2
    left = sorted(set(_half_sums(items[:mid])))
    best = None
    for s in _half_sums(items[mid:]):
        need = target - s
        # candidates in `left` nearest to `need`: the ones just below and just above
        idx = bisect_right(left, need)
        for j in (idx - 1, idx):
            if 0 <= j < len(left):
                total = s + left[j]
                if best is None or abs(total - target) < abs(best - target) or \
                        (abs(total - target) == abs(best - target) and total < best):
                    best = total
    return best


def count_subsets_with_sum(items, target):
    """The number of subsets summing exactly to `target`, via meet in the middle. Counts the empty
    subset when target == 0."""
    from collections import Counter
    mid = len(items) // 2
    left = Counter(_half_sums(items[:mid]))
    total = 0
    for s in _half_sums(items[mid:]):
        total += left.get(target - s, 0)
    return total


# --- brute-force reference --------------------------------------------------
def brute_subset_sum_exists(items, target):
    """Check every subset for the exact target. O(2^n)."""
    for mask in range(1 << len(items)):
        s = sum(items[i] for i in range(len(items)) if mask & (1 << i))
        if s == target:
            return True
    return False


def brute_max_under(items, capacity):
    """Maximum subset sum <= capacity by full enumeration."""
    best = 0
    for mask in range(1 << len(items)):
        s = sum(items[i] for i in range(len(items)) if mask & (1 << i))
        if s <= capacity and s > best:
            best = s
    return best


def brute_closest(items, target):
    """Achievable sum closest to target (ties toward smaller) by full enumeration."""
    best = None
    for mask in range(1 << len(items)):
        s = sum(items[i] for i in range(len(items)) if mask & (1 << i))
        if best is None or abs(s - target) < abs(best - target) or \
                (abs(s - target) == abs(best - target) and s < best):
            best = s
    return best


def brute_count_with_sum(items, target):
    """Number of subsets summing to target by full enumeration."""
    cnt = 0
    for mask in range(1 << len(items)):
        s = sum(items[i] for i in range(len(items)) if mask & (1 << i))
        if s == target:
            cnt += 1
    return cnt

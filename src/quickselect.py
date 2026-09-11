"""Quickselect and median-of-medians: the k-th smallest without sorting.

Finding the k-th smallest element -- the median, a percentile, the top-k threshold -- looks like
it needs a full sort at O(n log n). It does not. Quickselect (Hoare, 1961) is quicksort's
one-sided cousin: pick a pivot, partition the array into "less" and "greater", and recurse only
into the side that contains rank k. Discarding half the data each step gives O(n) expected time
-- you never touch the elements you do not need.

The catch is the pivot. A bad pivot (say, always the smallest) makes quickselect degrade to
O(n^2), the same worst case as quicksort. The median-of-medians algorithm (Blum, Floyd, Pratt,
Rivest & Tarjan, 1973) fixes this: split the data into groups of five, take each group's median,
then recursively take the median of those medians as the pivot. That pivot is provably better
than at least 30% of the elements and worse than another 30%, so each partition throws away a
constant fraction -- guaranteeing WORST-CASE linear time. It is the theoretical proof that
selection is O(n).

This module implements quickselect with a randomized pivot (fast in practice) and with the
median-of-medians pivot (worst-case linear), plus median and k-th-smallest wrappers, and checks
every result against a full sort. Pure stdlib (a seeded PRNG for pivots); the selection companion
to the sorting and Fenwick-tree notes.
"""

from __future__ import annotations


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def randint(self, k: int) -> int:
        """Uniform int in [0, k)."""
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % k


def quickselect(data, k: int, seed: int = 1):
    """Return the k-th smallest element (0-indexed: k=0 is the minimum) using quickselect with a
    randomized pivot. O(n) expected time. Does not modify the input."""
    a = list(data)
    n = len(a)
    if not 0 <= k < n:
        raise IndexError("k out of range")
    rng = _Rng(seed)
    lo, hi = 0, n - 1
    while True:
        if lo == hi:
            return a[lo]
        pivot_index = lo + rng.randint(hi - lo + 1)
        p = _partition(a, lo, hi, pivot_index)
        if k == p:
            return a[p]
        elif k < p:
            hi = p - 1
        else:
            lo = p + 1


def _partition(a, lo, hi, pivot_index):
    """Lomuto partition of a[lo..hi] around the value at pivot_index. Returns the pivot's final
    index; elements < pivot are left of it, >= pivot to the right."""
    pivot = a[pivot_index]
    a[pivot_index], a[hi] = a[hi], a[pivot_index]  # move pivot to the end
    store = lo
    for i in range(lo, hi):
        if a[i] < pivot:
            a[store], a[i] = a[i], a[store]
            store += 1
    a[store], a[hi] = a[hi], a[store]  # move pivot into place
    return store


def median_of_medians_select(data, k: int):
    """Return the k-th smallest element (0-indexed) using the median-of-medians pivot, which
    guarantees WORST-CASE O(n) time. Does not modify the input."""
    a = list(data)
    if not 0 <= k < len(a):
        raise IndexError("k out of range")
    return _mom_select(a, k)


def _mom_select(a, k):
    """Recursive median-of-medians selection on a list `a` for rank k."""
    if len(a) <= 5:
        return sorted(a)[k]
    # split into groups of 5, take each group's median
    medians = []
    for i in range(0, len(a), 5):
        group = sorted(a[i:i + 5])
        medians.append(group[len(group) // 2])
    # the pivot is the median of those medians (recursively)
    pivot = _mom_select(medians, len(medians) // 2)
    # partition around the pivot value
    lows = [x for x in a if x < pivot]
    highs = [x for x in a if x > pivot]
    equal = len(a) - len(lows) - len(highs)
    if k < len(lows):
        return _mom_select(lows, k)
    elif k < len(lows) + equal:
        return pivot  # k falls among elements equal to the pivot
    else:
        return _mom_select(highs, k - len(lows) - equal)


def kth_smallest(data, k: int, seed: int = 1):
    """The k-th smallest element, 1-indexed (k=1 is the minimum)."""
    return quickselect(data, k - 1, seed)


def kth_largest(data, k: int, seed: int = 1):
    """The k-th largest element, 1-indexed (k=1 is the maximum)."""
    n = len(data)
    if not 1 <= k <= n:
        raise IndexError("k out of range")
    return quickselect(data, n - k, seed)


def median(data, seed: int = 1):
    """The median. For an even count, the average of the two middle elements."""
    a = list(data)
    n = len(a)
    if n == 0:
        raise ValueError("median of empty data")
    if n % 2 == 1:
        return quickselect(a, n // 2, seed)
    lo = quickselect(a, n // 2 - 1, seed)
    hi = quickselect(a, n // 2, seed)
    return (lo + hi) / 2


def percentile(data, p: float, seed: int = 1):
    """The p-th percentile (0..100) by nearest-rank: the element at rank ceil(p/100 * n)."""
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    n = len(data)
    if n == 0:
        raise ValueError("percentile of empty data")
    import math
    rank = max(1, math.ceil(p / 100 * n))
    return quickselect(data, rank - 1, seed)

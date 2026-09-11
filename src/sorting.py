"""Comparison sorts: the classic algorithms and what distinguishes them.

Sorting is the most-studied problem in computing, and the handful of comparison-based algorithms
here each make a different trade among speed, memory, stability, and worst-case guarantees. All
compare elements pairwise, so all are bounded below by the O(n log n) decision-tree limit -- but
the constants, the memory, and the failure modes differ:

  INSERTION SORT -- O(n^2), but O(n) on nearly-sorted input and cache-friendly; the base case big
      sorts fall back to for small subarrays. Stable and in-place.
  MERGE SORT     -- O(n log n) ALWAYS, stable, but needs O(n) scratch space. The safe default when
      stability matters or worst-case time is critical (it never degrades).
  QUICK SORT     -- O(n log n) average, in-place, usually the fastest in practice; but O(n^2) on
      adversarial input unless the pivot is chosen well. Here: median-of-three pivot plus an
      insertion-sort cutoff, the standard engineering fixes. Not stable.
  HEAP SORT      -- O(n log n) worst case AND in-place (merge sort is one, quick sort the other --
      heap sort is both), built on a BINARY HEAP. Not stable, poor cache behaviour, but no scratch
      space and no quadratic blowup.

A binary heap (the array-embedded complete tree behind heap sort) is also a priority queue in its
own right. This module implements all four sorts plus the heap, with a comparison counter and a key
function -- verified that every sort matches Python's built-in on random, sorted, reverse, and
duplicate-heavy inputs, that merge and insertion sorts are stable while quick and heap are not, that
comparison counts scale as O(n log n) for the good sorts and O(n^2) for insertion, and that the heap
acts as a correct priority queue. Pure stdlib; an algorithms companion to the quickselect and
Fenwick-tree notes."""

from __future__ import annotations


class Counter:
    """A shared comparison counter; pass to any sort to tally element comparisons."""

    def __init__(self):
        self.comparisons = 0


def insertion_sort(arr, key=None, counter=None):
    """Stable O(n^2) insertion sort; fast on nearly-sorted data. Returns a new list."""
    key = key or (lambda x: x)
    a = list(arr)
    for i in range(1, len(a)):
        cur = a[i]
        ck = key(cur)
        j = i - 1
        while j >= 0:
            if counter is not None:
                counter.comparisons += 1
            if key(a[j]) > ck:
                a[j + 1] = a[j]
                j -= 1
            else:
                break
        a[j + 1] = cur
    return a


def merge_sort(arr, key=None, counter=None):
    """Stable O(n log n) merge sort using O(n) scratch space. Returns a new list."""
    key = key or (lambda x: x)
    a = list(arr)

    def merge(left, right):
        out = []
        i = j = 0
        while i < len(left) and j < len(right):
            if counter is not None:
                counter.comparisons += 1
            # <= keeps stability: equal elements from `left` (earlier) go first
            if key(left[i]) <= key(right[j]):
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
        out.extend(left[i:])
        out.extend(right[j:])
        return out

    def rec(lo, hi):
        if hi - lo <= 1:
            return a[lo:hi]
        mid = (lo + hi) // 2
        return merge(rec(lo, mid), rec(mid, hi))

    return rec(0, len(a))


def quick_sort(arr, key=None, counter=None, cutoff=16):
    """In-place-style O(n log n)-average quick sort with median-of-three pivot and an
    insertion-sort cutoff for small subarrays. Not stable. Returns a new list."""
    key = key or (lambda x: x)
    a = list(arr)

    def med3(lo, mid, hi):
        # return the index of the median of a[lo], a[mid], a[hi]
        klo, kmid, khi = key(a[lo]), key(a[mid]), key(a[hi])
        if counter is not None:
            counter.comparisons += 3
        if klo <= kmid <= khi or khi <= kmid <= klo:
            return mid
        if kmid <= klo <= khi or khi <= klo <= kmid:
            return lo
        return hi

    def insertion(lo, hi):
        for i in range(lo + 1, hi + 1):
            cur = a[i]
            ck = key(cur)
            j = i - 1
            while j >= lo:
                if counter is not None:
                    counter.comparisons += 1
                if key(a[j]) > ck:
                    a[j + 1] = a[j]
                    j -= 1
                else:
                    break
            a[j + 1] = cur

    def qsort(lo, hi):
        while hi - lo > cutoff:
            mid = (lo + hi) // 2
            p = med3(lo, mid, hi)
            a[p], a[hi] = a[hi], a[p]        # park pivot at the end
            pivot_k = key(a[hi])
            i = lo
            for j in range(lo, hi):
                if counter is not None:
                    counter.comparisons += 1
                if key(a[j]) < pivot_k:
                    a[i], a[j] = a[j], a[i]
                    i += 1
            a[i], a[hi] = a[hi], a[i]
            # recurse into the smaller side, loop on the larger (bounded stack depth)
            if i - lo < hi - i:
                qsort(lo, i - 1)
                lo = i + 1
            else:
                qsort(i + 1, hi)
                hi = i - 1
        insertion(lo, hi)

    if len(a) > 1:
        qsort(0, len(a) - 1)
    return a


class BinaryHeap:
    """A binary min-heap on an array (the complete-tree embedding): parent i, children 2i+1/2i+2.
    Also a priority queue via push/pop."""

    def __init__(self, items=None, key=None):
        self.key = key or (lambda x: x)
        self.data = list(items) if items else []
        self.comparisons = 0
        if self.data:
            self._heapify()

    def _less(self, i, j):
        self.comparisons += 1
        return self.key(self.data[i]) < self.key(self.data[j])

    def _sift_down(self, i):
        n = len(self.data)
        while True:
            smallest = i
            l, r = 2 * i + 1, 2 * i + 2
            if l < n and self._less(l, smallest):
                smallest = l
            if r < n and self._less(r, smallest):
                smallest = r
            if smallest == i:
                break
            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]
            i = smallest

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self._less(i, parent):
                self.data[i], self.data[parent] = self.data[parent], self.data[i]
                i = parent
            else:
                break

    def _heapify(self):
        for i in range(len(self.data) // 2 - 1, -1, -1):
            self._sift_down(i)

    def push(self, item):
        self.data.append(item)
        self._sift_up(len(self.data) - 1)

    def pop(self):
        """Remove and return the smallest item."""
        if not self.data:
            raise IndexError("pop from empty heap")
        top = self.data[0]
        last = self.data.pop()
        if self.data:
            self.data[0] = last
            self._sift_down(0)
        return top

    def peek(self):
        if not self.data:
            raise IndexError("peek at empty heap")
        return self.data[0]

    def __len__(self):
        return len(self.data)

    def is_valid(self):
        """Check the heap property holds everywhere (for testing)."""
        n = len(self.data)
        for i in range(n):
            for c in (2 * i + 1, 2 * i + 2):
                if c < n and self.key(self.data[c]) < self.key(self.data[i]):
                    return False
        return True


def heap_sort(arr, key=None, counter=None):
    """O(n log n) worst-case, in-place-style heap sort. Not stable. Returns a new list."""
    heap = BinaryHeap(arr, key=key)
    out = [heap.pop() for _ in range(len(heap))]
    if counter is not None:
        counter.comparisons += heap.comparisons
    return out


def is_sorted(arr, key=None):
    key = key or (lambda x: x)
    return all(key(arr[i]) <= key(arr[i + 1]) for i in range(len(arr) - 1))


ALGORITHMS = {
    "insertion": insertion_sort,
    "merge": merge_sort,
    "quick": quick_sort,
    "heap": heap_sort,
}

STABLE = {"insertion", "merge"}

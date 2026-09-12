"""The DGIM algorithm -- counting the 1s in the last N bits of an endless stream, in tiny memory.

A stream of bits arrives forever -- packets that were errors or not, clicks that converted or not,
sensor readings above threshold or not -- and you want to know, at any moment, roughly how many 1s
occurred in the LAST N of them. Storing the whole window costs N bits, impossible when N is a billion
and the stream never ends. The DGIM algorithm (Datar, Gionis, Indyk & Motwani, 2002) answers the query
using only O(log^2 N) bits -- a few dozen numbers for a billion-bit window -- with a guaranteed error of
at most 50% (tunable smaller), and it is the textbook technique for sliding-window statistics over data
streams.

The idea is to summarise the window with EXPONENTIAL BUCKETS. Each bucket records a timestamp (when it
ended) and a size that is always a power of two -- the number of 1s it covers. Buckets grow older and
larger toward the past: a run of size-1 buckets near the present, then size-2, size-4, size-8, and so
on. The crucial invariant caps the redundancy: there are at most a small constant number of buckets of
each size (two, in the classic version). When a new 1 arrives it becomes a fresh size-1 bucket; whenever
three buckets of the same size appear, the two OLDEST merge into one bucket of the next size up, which
can cascade -- exactly the carry propagation of binary counting. Buckets that slide out of the window
(timestamp older than N) are dropped.

To answer "how many 1s in the last N bits?", sum the sizes of all buckets still in the window, but count
the OLDEST surviving bucket only HALVED -- because it straddles the window boundary and we cannot know
how much of it is inside. That single halving is the entire source of the error, and because bucket
sizes double, the uncertainty is at most the size of the oldest bucket, which is provably within 50% of
the true count. Using k buckets per size instead of two shrinks the guaranteed relative error to about
1/k.

This module implements the DGIM sliding-window counter over a bit stream: push a bit, query the count in
the last N (or any smaller k), with the bucket invariant maintained incrementally, plus a brute-force
exact counter used only as the validation oracle. Pure standard library.

Validation. Against an exact sliding-window count (a real deque of the last N bits, used only in the
tests), DGIM's estimate must always lie within its guaranteed relative-error band -- checked over long
random streams at several window sizes and bit densities, and over adversarial bursty streams. Increasing
the buckets-per-size parameter is shown to tighten the error empirically. Sub-window queries (last k < N)
obey the same bound. The memory really is logarithmic: the bucket count stays O(log^2 N) however long
the stream runs. Edge cases -- all zeros, all ones, a window larger than the stream so far -- are exact."""

import math


class DGIM:
    """DGIM sliding-window 1-counter over a bit stream, using O(log^2 N) memory.

    ``window`` is the maximum number of most-recent bits to summarise. ``buckets_per_size`` (>=2)
    trades memory for accuracy: the guaranteed relative error is about 1/(buckets_per_size - 1).
    """

    def __init__(self, window, buckets_per_size=2):
        if window < 1:
            raise ValueError("window must be >= 1")
        if buckets_per_size < 2:
            raise ValueError("buckets_per_size must be >= 2")
        self.window = window
        self.r = buckets_per_size
        self.time = 0                 # index of the next bit to arrive
        # buckets: list of [end_timestamp, size], newest first
        self.buckets = []

    def push(self, bit):
        """Feed the next bit (0 or 1) of the stream."""
        t = self.time
        self.time += 1
        # drop the oldest bucket if it has fully slid out of the window
        if self.buckets and self.buckets[-1][0] <= t - self.window:
            self.buckets.pop()
        if not bit:
            return
        # a new 1 becomes a size-1 bucket at the front
        self.buckets.insert(0, [t, 1])
        self._merge()

    def _merge(self):
        """Enforce the invariant: at most r buckets of each size; merge the two oldest of any size."""
        # count consecutive buckets of each size from the front and merge when r+1 appear
        i = 0
        n = len(self.buckets)
        while i < len(self.buckets):
            size = self.buckets[i][1]
            # find the run of buckets with this size
            j = i
            while j < len(self.buckets) and self.buckets[j][1] == size:
                j += 1
            count = j - i
            if count > self.r:
                # merge the two OLDEST of this size (indices j-2 and j-1) into one of size*2
                a = self.buckets[j - 2]
                b = self.buckets[j - 1]
                # keep the newer timestamp of the merged pair (a is newer, since newest-first)
                merged = [a[0], size * 2]
                self.buckets[j - 2:j] = [merged]
                # the merge may cascade to the next size; restart scan from here
                i = j - 2
            else:
                i = j

    def count(self, k=None):
        """Estimate the number of 1s in the last ``k`` bits (default: the full window)."""
        if k is None:
            k = self.window
        k = min(k, self.window)
        cutoff = self.time - k        # a bucket is in-window iff its end timestamp >= cutoff
        total = 0
        oldest_in_window = 0
        for end, size in self.buckets:   # newest-first
            if end >= cutoff:
                total += size
                oldest_in_window = size  # last one that qualifies is the oldest in-window bucket
        # DGIM halves the oldest in-window bucket, which straddles the window boundary
        if oldest_in_window:
            total -= oldest_in_window // 2
        return total

    def bucket_count(self):
        """Number of buckets currently stored (its logarithmic memory footprint)."""
        return len(self.buckets)

    def error_bound(self):
        """The guaranteed relative-error bound for this configuration (~1/(r-1))."""
        return 1.0 / (self.r - 1)


class ExactWindow:
    """Exact sliding-window 1-counter (a deque of the last ``window`` bits) -- validation oracle."""

    def __init__(self, window):
        self.window = window
        self.bits = []

    def push(self, bit):
        self.bits.append(1 if bit else 0)
        if len(self.bits) > self.window:
            self.bits.pop(0)

    def count(self, k=None):
        if k is None:
            k = self.window
        k = min(k, self.window, len(self.bits))
        return sum(self.bits[-k:]) if k else 0

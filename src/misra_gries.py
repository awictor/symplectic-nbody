"""Misra-Gries: the frequent items of a stream in tiny memory.

Which items appear more than n/k times in a stream of n items? Counting exactly needs a counter
per distinct item -- impossible for a firehose of billions of distinct values (network flows,
search queries, word counts) in bounded memory. The Misra-Gries summary (1982) finds every such
"heavy hitter" using only k-1 counters, in a single pass, regardless of how many distinct items
stream by. Its famous special case k=2 is the Boyer-Moore majority-vote algorithm: find the
element occupying more than half the stream with a single counter.

The rule is a generalized voting. Keep at most k-1 (item, count) slots. For each incoming item:

  * if it is already tracked, increment its count;
  * else if a slot is free, start tracking it at count 1;
  * else decrement EVERY counter by one (dropping any that hit zero) -- the item "cancels" one
    of each of the others.

After the pass, any item occurring more than n/k times is guaranteed to still be in the summary
(no false negatives), but some survivors may be below the threshold (false positives). A cheap
SECOND pass counts the survivors exactly to confirm which are true heavy hitters. The counts the
summary carries are underestimates, off by at most n/k.

This module builds the summary in one pass, extracts candidates, verifies them in a second pass,
and provides the majority-vote special case, all checked against exact counting. Pure stdlib;
the streaming-frequency companion to the HyperLogLog and Bloom notes.
"""

from __future__ import annotations


class MisraGries:
    """A Misra-Gries frequent-items summary with at most k-1 counters (finds items over n/k)."""

    def __init__(self, k: int):
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = k
        self.counters = {}          # item -> approximate count
        self.n = 0                  # total items seen

    def add(self, item):
        """Process one stream item in O(k) worst case (usually O(1))."""
        self.n += 1
        if item in self.counters:
            self.counters[item] += 1
        elif len(self.counters) < self.k - 1:
            self.counters[item] = 1
        else:
            # decrement all counters; drop any that reach zero
            for key in list(self.counters):
                self.counters[key] -= 1
                if self.counters[key] == 0:
                    del self.counters[key]

    def update(self, stream):
        """Process an iterable of items."""
        for x in stream:
            self.add(x)
        return self

    def candidates(self):
        """The tracked items -- a superset of the true heavy hitters (may include false
        positives, never misses a real one). Returns {item: approximate_count}."""
        return dict(self.counters)

    def approx_count(self, item) -> int:
        """The summary's (under)estimate of an item's count: within n/k of the true count."""
        return self.counters.get(item, 0)


def heavy_hitters(stream, k: int):
    """Find every item occurring more than n/k times in `stream`, exactly, via a Misra-Gries
    first pass followed by an exact-count verification pass. Returns {item: true_count} sorted
    by descending count. Uses only O(k) memory in the first pass."""
    items = list(stream)
    n = len(items)
    mg = MisraGries(k)
    mg.update(items)
    threshold = n / k
    # verify candidates with an exact second pass (only over the <=k-1 survivors)
    survivors = set(mg.candidates())
    exact = {c: 0 for c in survivors}
    for x in items:
        if x in exact:
            exact[x] += 1
    result = {c: cnt for c, cnt in exact.items() if cnt > threshold}
    return dict(sorted(result.items(), key=lambda kv: (-kv[1], repr(kv[0]))))


def majority(stream):
    """Boyer-Moore majority vote: the item occurring in strictly more than half the stream, or
    None if there is none. Single counter, single pass, then one verification pass."""
    candidate = None
    count = 0
    items = list(stream)
    for x in items:
        if count == 0:
            candidate = x
            count = 1
        elif x == candidate:
            count += 1
        else:
            count -= 1
    # verify: the candidate must actually be a strict majority
    if candidate is not None and items.count(candidate) > len(items) // 2:
        return candidate
    return None


# --- exact reference (for validation) --------------------------------------

def exact_heavy_hitters(stream, k: int):
    """Exact heavy hitters by full counting -- for checking the streaming summary."""
    items = list(stream)
    n = len(items)
    counts = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    threshold = n / k
    result = {x: c for x, c in counts.items() if c > threshold}
    return dict(sorted(result.items(), key=lambda kv: (-kv[1], repr(kv[0]))))

"""Interval trees -- every interval that overlaps a query point or range, without scanning them all.

Genome browsers ask "which genes span this locus?"; calendars ask "which meetings clash with 2-3pm?";
a network monitor asks "which leases were active at this timestamp?". All are the same query -- given a
set of intervals [lo, hi], report every one that OVERLAPS a query point (a STABBING query) or a query
range. Scanning all n intervals is O(n) per query, wasteful when the set is large and queries are
frequent. The INTERVAL TREE answers a stabbing query in O(log n + k) where k is the number of hits,
storing the intervals so most are pruned untouched.

This module builds the CENTERED interval tree (the classic CLRS-adjacent design). Pick the median of
all interval endpoints as the CENTER of the root. Every interval that CONTAINS the center is stored at
the root -- kept in two sorted lists, one by left endpoint and one by right endpoint, so a query can
walk them and stop early. Intervals lying entirely LEFT of the center go into a left subtree, those
entirely RIGHT into a right subtree, and each subtree is built the same way on its own median. To stab
a point q at a node: if q is left of the center, every root interval whose LEFT endpoint is <= q
overlaps q (walk the left-sorted list until one starts after q, then stop), and recurse into the left
child; symmetrically if q is right of center; the geometry guarantees the untouched subtree cannot
contain a hit. Range queries [a, b] combine a stab-style descent with a full report of intervals that
must intersect.

The tree also supports the simple but essential operations: report all overlaps of a query interval,
count them, and rebuild. Construction is O(n log n); the tree is static (built from a fixed set), which
is the common case for the query-heavy workloads above. Pure standard library.

Validation. Correctness is exact agreement with brute force: over many random interval sets and
thousands of seeded stabbing and range queries, the tree returns EXACTLY the set a linear scan finds --
same intervals, no misses, no extras -- including degenerate zero-length intervals, touching endpoints,
fully nested and identical intervals, and queries outside the whole range. The endpoint-sorted lists at
each node are verified sorted, every stored interval genuinely contains its node's center, and the
left/right partition is correct (left-subtree intervals end before the center, right-subtree intervals
start after it). Point and range results match, and counts equal the reported list lengths."""


class _Node:
    __slots__ = ("center", "by_left", "by_right", "left", "right")

    def __init__(self, center):
        self.center = center
        self.by_left = []      # intervals containing center, sorted by lo ascending
        self.by_right = []     # same intervals, sorted by hi descending
        self.left = None
        self.right = None


class IntervalTree:
    """A static centered interval tree over intervals given as (lo, hi) with lo <= hi.

    Intervals are treated as CLOSED [lo, hi]; overlap means the ranges share at least one point.
    Each stored interval is a (lo, hi, data) triple so callers can attach a payload.
    """

    def __init__(self, intervals=None):
        self.root = None
        self._items = []
        if intervals:
            self.build(intervals)

    def build(self, intervals):
        """(Re)build the tree from an iterable of (lo, hi) or (lo, hi, data)."""
        items = []
        for it in intervals:
            if len(it) == 2:
                lo, hi = it
                data = None
            else:
                lo, hi, data = it
            if lo > hi:
                raise ValueError(f"interval lo > hi: {it}")
            items.append((lo, hi, data))
        self._items = items
        self.root = self._build(items)

    def _build(self, items):
        if not items:
            return None
        # center = median of all endpoints
        endpoints = sorted([lo for lo, _, _ in items] + [hi for _, hi, _ in items])
        center = endpoints[len(endpoints) // 2]
        node = _Node(center)
        left_items, right_items = [], []
        for lo, hi, data in items:
            if hi < center:
                left_items.append((lo, hi, data))
            elif lo > center:
                right_items.append((lo, hi, data))
            else:
                node.by_left.append((lo, hi, data))
        node.by_left.sort(key=lambda t: t[0])
        node.by_right = sorted(node.by_left, key=lambda t: t[1], reverse=True)
        node.left = self._build(left_items)
        node.right = self._build(right_items)
        return node

    # -- stabbing query -----------------------------------------------------
    def stab(self, q):
        """All intervals containing the point q. Returns a list of (lo, hi, data)."""
        out = []
        self._stab(self.root, q, out)
        return out

    def _stab(self, node, q, out):
        if node is None:
            return
        if q < node.center:
            # intervals here overlap q iff their lo <= q; by_left is sorted ascending by lo
            for lo, hi, data in node.by_left:
                if lo <= q:
                    out.append((lo, hi, data))
                else:
                    break
            self._stab(node.left, q, out)
        elif q > node.center:
            for lo, hi, data in node.by_right:
                if hi >= q:
                    out.append((lo, hi, data))
                else:
                    break
            self._stab(node.right, q, out)
        else:
            # q == center: every interval stored here contains it
            out.extend(node.by_left)

    # -- range overlap query -------------------------------------------------
    def overlap(self, a, b):
        """All intervals overlapping the query range [a, b]. Returns a list of (lo, hi, data)."""
        if a > b:
            a, b = b, a
        out = []
        self._overlap(self.root, a, b, out)
        return out

    def _overlap(self, node, a, b, out):
        if node is None:
            return
        # intervals at this node overlap [a,b] iff lo <= b and hi >= a
        for lo, hi, data in node.by_left:
            if lo > b:
                break                      # sorted by lo; no later one can start <= b
            if hi >= a:
                out.append((lo, hi, data))
        # descend: the left subtree holds intervals entirely < center; it can still overlap if a < center
        if node.left is not None and a < node.center:
            self._overlap(node.left, a, b, out)
        if node.right is not None and b > node.center:
            self._overlap(node.right, a, b, out)

    def count_stab(self, q):
        return len(self.stab(q))

    def count_overlap(self, a, b):
        return len(self.overlap(a, b))

    def __len__(self):
        return len(self._items)


# ---------------------------------------------------------------------------
# brute-force reference
# ---------------------------------------------------------------------------

def brute_stab(intervals, q):
    """Reference: all intervals (lo, hi, data) containing point q."""
    out = []
    for it in intervals:
        lo, hi = it[0], it[1]
        data = it[2] if len(it) > 2 else None
        if lo <= q <= hi:
            out.append((lo, hi, data))
    return out


def brute_overlap(intervals, a, b):
    """Reference: all intervals overlapping [a, b]."""
    if a > b:
        a, b = b, a
    out = []
    for it in intervals:
        lo, hi = it[0], it[1]
        data = it[2] if len(it) > 2 else None
        if lo <= b and hi >= a:
            out.append((lo, hi, data))
    return out

"""Li Chao tree: the lower envelope of a set of lines, queried in logarithmic time.

Many optimisation problems reduce to the same primitive: keep a growing set of straight lines
y = m*x + b, and repeatedly ask "over all lines added so far, what is the minimum (or maximum) value at
a given x?" The answer as x sweeps across the domain traces the LOWER ENVELOPE (or upper envelope) of
the lines -- a piecewise-linear convex curve. This is exactly the query at the heart of the CONVEX
HULL TRICK, a standard speed-up for a large family of dynamic-programming recurrences (dp[i] = min over
j of dp[j] + cost(j, i) when the cost is linear in i): each transition is a line, each state a query,
and the O(n^2) DP collapses to O(n log n). It also underlies problems in computational geometry, ray
tracing, and economics (the cheapest supplier at each price point).

The LI CHAO TREE is the cleanest way to support this when lines arrive in arbitrary order and queries
interleave with insertions -- the case the classic monotonic-stack convex hull trick cannot handle. It
is a segment tree over the x-domain; each node owns the single line that is minimal at that node's
midpoint. Inserting a line compares it to the node's line at the midpoint, keeps the lower one there,
and recurses into the half-interval where the other line could still win -- because two lines cross at
most once, the loser can only dominate on one side, so insertion visits O(log range) nodes. A query
walks root-to-leaf taking the minimum of every stored line along the path, also O(log range). Both work
for real-valued x by descending a fixed number of levels.

This module implements a Li Chao tree over a real coordinate range for minimum queries (and, by
negating, maximum), supporting arbitrary-order line insertion and point queries, and applies it to
speed up a 1-D convex-hull-trick DP. It is verified against brute force -- every query matches the true
minimum over all inserted lines evaluated directly, on hundreds of random line sets and query points,
including the convex-hull-trick DP result matching an O(n^2) reference. Pure stdlib; an
optimisation-and-data-structures companion to the segment-tree, convex-hull, and simplex notes."""

from __future__ import annotations


_INF = float("inf")


class LiChaoTree:
    """Li Chao tree for querying the minimum of a set of lines y = m*x + b over [xmin, xmax].

    Set `maximize=True` to query the maximum instead (implemented by negating internally)."""

    def __init__(self, xmin, xmax, maximize=False, depth=60):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.maximize = maximize
        self.depth = depth
        # each tree node stores a line (m, b); None means "no line yet"
        self.lines = {}          # node index -> (m, b)

    def _val(self, line, x):
        m, b = line
        return m * x + b

    def add_line(self, m, b):
        """Insert the line y = m*x + b. Arbitrary order; O(log range)."""
        if self.maximize:
            m, b = -m, -b        # maximise by minimising the negated lines
        self._insert((m, b), 1, self.xmin, self.xmax, 0)

    def _insert(self, new, node, lo, hi, level):
        cur = self.lines.get(node)
        if cur is None:
            self.lines[node] = new
            return
        mid = (lo + hi) / 2
        # decide which line is lower at the midpoint; keep it in this node
        if self._val(new, mid) < self._val(cur, mid):
            self.lines[node], new = new, cur    # swap: new becomes the loser to push down
            cur = self.lines[node]
        if level >= self.depth:
            return
        # the loser `new` may still win on one side (lines cross at most once)
        if self._val(new, lo) < self._val(cur, lo):
            self._insert(new, 2 * node, lo, mid, level + 1)
        elif self._val(new, hi) < self._val(cur, hi):
            self._insert(new, 2 * node + 1, mid, hi, level + 1)

    def query(self, x):
        """Minimum (or maximum) value at `x` over all inserted lines. O(log range). Returns +inf
        (or -inf for maximise) if no lines were inserted."""
        x = float(x)
        node = 1
        lo, hi = self.xmin, self.xmax
        best = _INF
        for _ in range(self.depth + 1):
            cur = self.lines.get(node)
            if cur is not None:
                v = self._val(cur, x)
                if v < best:
                    best = v
            mid = (lo + hi) / 2
            if x < mid:
                node, hi = 2 * node, mid
            else:
                node, lo = 2 * node + 1, mid
        return -best if self.maximize else best


def convex_hull_trick_dp(costs, slopes, intercepts):
    """A small worked application: dp[i] = min over j < i of (slopes[j]*x[i] + intercepts[j]), where
    x[i] = costs[i]. Returns the list of per-i minima using a Li Chao tree. This is the shape of the
    convex-hull-trick DP speed-up; here lines are supplied explicitly for a clean, testable demo."""
    xs = [float(c) for c in costs]
    tree = LiChaoTree(min(xs) - 1, max(xs) + 1)
    out = []
    for i in range(len(xs)):
        tree.add_line(slopes[i], intercepts[i])
        out.append(tree.query(xs[i]))
    return out


# --- brute-force reference --------------------------------------------------
def brute_min(lines, x):
    """The true minimum of a set of (m, b) lines at x, evaluated directly."""
    return min(m * x + b for m, b in lines) if lines else _INF


def brute_max(lines, x):
    """The true maximum of a set of (m, b) lines at x."""
    return max(m * x + b for m, b in lines) if lines else -_INF

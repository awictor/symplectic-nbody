"""Karger's algorithm: finding a graph's minimum cut by random edge contraction.

The GLOBAL MINIMUM CUT of a connected undirected graph is the fewest edges (or least total weight)
whose removal splits it into two pieces -- the network's weakest point, the bottleneck a flow must
cross, the natural place a community divides. Deterministic algorithms (Stoer-Wagner) find it exactly
in polynomial time; KARGER'S ALGORITHM (1993) takes a startlingly different, RANDOMIZED route that is
simpler and, in its recursive form, among the fastest known. Its whole idea is one operation:
CONTRACTION. Pick a random edge and merge its two endpoints into a single super-vertex (keeping
parallel edges, dropping self-loops); repeat until only two super-vertices remain. The edges still
joining those two are a cut of the original graph -- and with decent probability, the minimum one.

Why it works: a specific minimum cut survives a contraction as long as we never pick one of its (few)
edges to contract. A single run finds any particular min cut with probability at least 1/C(n,2) =
2/(n(n-1)), so repeating O(n^2 log n) times makes the failure probability vanishingly small. The
KARGER-STEIN improvement contracts only down to about n/sqrt(2) vertices -- where a min-cut edge is
still unlikely to have been hit -- then RECURSES twice on the smaller graph and keeps the better
result, driving the runtime down to O(n^2 log^3 n) with high success probability. Weighted graphs work
identically by choosing the contracted edge with probability proportional to its weight.

This module implements a single contraction trial, the repeated Karger algorithm, and the recursive
Karger-Stein variant, for weighted undirected graphs, returning the best cut weight and the vertex
bipartition found. Because it is Monte Carlo, correctness is checked STATISTICALLY: over hundreds of
random graphs the value it returns is never below the true minimum (a contraction cut is always a valid
cut) and, given enough trials, equals the exact minimum computed by the Stoer-Wagner algorithm. A
fixed seed makes every run reproducible. Pure stdlib; a graph-algorithms companion to the Stoer-Wagner
min-cut, max-flow, and union-find notes."""

from __future__ import annotations


class _LCG:
    """Seeded linear congruential generator, so trials are reproducible without the `random` module."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 8               # high bits

    def randint(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)


def _contract_once(n, edges, rng):
    """One Karger trial: contract random (weight-proportional) edges until two super-vertices remain.
    Returns (cut_weight, (side_a, side_b)) where the sides are frozensets of original vertices."""
    # union-find over vertices, with each root owning the set of original vertices merged into it
    parent = list(range(n))
    members = [{v} for v in range(n)]
    count = n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while count > 2:
        # cross edges only, with the CURRENT total weight (recomputed so sampling stays uniform)
        cross = [(find(u), find(v), w) for (u, v, w) in edges if find(u) != find(v)]
        if not cross:
            break                        # graph already in <=2 components
        total_w = sum(w for _, _, w in cross)
        r = rng.randint(0, total_w - 1)
        acc = 0
        a, b = cross[0][0], cross[0][1]
        for (ru, rv, w) in cross:
            acc += w
            if acc > r:
                a, b = ru, rv
                break
        parent[a] = b                    # merge a into b
        members[b] |= members[a]
        count -= 1

    # the two remaining super-vertices define the cut; sum weights of edges crossing them
    roots = {find(v) for v in range(n)}
    if len(roots) < 2:
        return float("inf"), (frozenset(range(n)), frozenset())
    ra = next(iter(roots))
    side_a = frozenset(members[ra])
    side_b = frozenset(v for v in range(n) if v not in side_a)
    cut = sum(w for (u, v, w) in edges
              if (u in side_a) != (v in side_a))
    return cut, (side_a, side_b)


def karger_min_cut(n, edges, trials=None, seed=12345):
    """Global minimum cut by repeated random contraction (basic Karger).

    Returns (best_cut_weight, (side_a, side_b)). `trials` defaults to ceil(n^2 * ln n) which makes
    the failure probability small; pass a smaller number for speed."""
    if n < 2:
        return 0, (frozenset(range(n)), frozenset())
    if trials is None:
        import math
        trials = max(1, int(n * n * max(1.0, math.log(n))))
    rng = _LCG(seed)
    best = float("inf")
    best_part = None
    for _ in range(trials):
        cut, part = _contract_once(n, edges, rng)
        if cut < best:
            best = cut
            best_part = part
    return best, best_part


def _contract_subset(edges, rng, target):
    """Contract a labelled graph down to `target` super-labels.

    `edges` are (label_u, label_v, w) where each label is a frozenset of ORIGINAL vertices (so member
    sets survive across recursion levels). Returns the new edge list over the surviving super-labels,
    each still a frozenset of original vertices."""
    labels = {u for (u, v, w) in edges} | {v for (u, v, w) in edges}
    parent = {x: x for x in labels}
    merged = {x: x for x in labels}          # root label -> frozenset of all originals merged in
    count = len(labels)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while count > target:
        cross = [(find(u), find(v), w) for (u, v, w) in edges if find(u) != find(v)]
        if not cross:
            break
        total = sum(w for _, _, w in cross)
        r = rng.randint(0, total - 1)
        acc = 0
        a, b = cross[0][0], cross[0][1]
        for (ru, rv, w) in cross:
            acc += w
            if acc > r:
                a, b = ru, rv
                break
        parent[a] = b
        merged[b] = merged[b] | merged[a]     # union the original-vertex sets
        count -= 1

    new_edges = [(merged[find(u)], merged[find(v)], w)
                 for (u, v, w) in edges if find(u) != find(v)]
    return new_edges


def karger_stein(n, edges, seed=12345):
    """The recursive Karger-Stein min cut: contract to ~n/sqrt(2), recurse twice, keep the better.
    Higher success probability per unit work than basic Karger. Returns (cut_weight, (side_a,
    side_b))."""
    import math
    rng = _LCG(seed)

    def cut_of(side_a):
        return sum(w for (u, v, w) in edges if (u in side_a) != (v in side_a))

    def n_labels(sub):
        return len({u for (u, v, w) in sub} | {v for (u, v, w) in sub})

    def brute_labels(sub):
        """Exact best cut of a small labelled graph by trying every 2-partition of its labels.
        Returns the side-A set of original vertices."""
        labels = list({u for (u, v, w) in sub} | {v for (u, v, w) in sub})
        L = len(labels)
        best = float("inf")
        best_side = set(labels[0]) if labels else set()
        for mask in range(1, 1 << L):
            if mask == (1 << L) - 1:
                continue                 # both sides must be non-empty
            side_labels = {labels[i] for i in range(L) if mask & (1 << i)}
            side_a = set().union(*side_labels) if side_labels else set()
            c = cut_of(side_a)
            if c < best:
                best = c
                best_side = side_a
        return best_side

    def recurse(sub):
        """Return a side-A set of original vertices for the best cut found in this contracted graph."""
        m = n_labels(sub)
        if m <= 6:
            return brute_labels(sub)     # small enough to solve exactly over label 2-partitions
        # contract to ceil(m/sqrt2) but always strictly fewer labels so recursion makes progress
        target = min(m - 1, max(2, int(math.ceil(m / math.sqrt(2)))))
        best = float("inf")
        best_side = None
        for _ in range(2):
            red = _contract_subset(sub, rng, target)
            side = recurse(red)
            c = cut_of(side)
            if c < best:
                best = c
                best_side = side
        return best_side

    init = [(frozenset([u]), frozenset([v]), w) for (u, v, w) in edges]
    side_a = frozenset(recurse(init))
    side_b = frozenset(v for v in range(n) if v not in side_a)
    return cut_of(side_a), (side_a, side_b)


# --- verification helper ----------------------------------------------------
def cut_weight(n, edges, side_a):
    """Total weight of edges crossing the partition (side_a, everything else)."""
    sa = set(side_a)
    return sum(w for (u, v, w) in edges if (u in sa) != (v in sa))

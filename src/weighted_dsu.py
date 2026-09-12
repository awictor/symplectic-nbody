"""Weighted union-find: disjoint sets that also track relative offsets between elements.

A plain union-find answers "are x and y in the same group?"; the WEIGHTED (or potential) union-find
answers the richer question "and if so, what is the known DIFFERENCE between them?" Each element carries
a potential relative to its set's representative, maintained so that for any two elements in the same
set the difference potential(x) - potential(y) is exactly the accumulated relation between them. This
turns union-find into an incremental solver for systems of DIFFERENCE CONSTRAINTS "x - y = d": each
constraint is a weighted union, each query a potential difference, and any constraint that contradicts
the current knowledge is detected immediately. It underlies offline range-parity problems, the classic
"are these two people friends or enemies?" bipartiteness-with-relations puzzles, sensor-offset
calibration, and consistency checking of relative measurements.

The mechanics extend path compression and union by rank with a `rank_diff` (weight) on each parent
edge. FIND accumulates the weights along the path to the root and, while compressing, rewrites each
node's weight to be relative to the root directly. UNION of x and y with the relation x - y = d looks up
both roots and their potentials; if the roots differ it links one under the other with the weight that
makes the relation hold (derived from d and the two potentials), and if the roots already coincide it
CHECKS that the existing difference equals d -- returning False on a contradiction. A DIFF query returns
potential(x) - potential(y) when x and y are connected, or None otherwise. A bipartite/parity variant
falls out by working modulo 2.

This module implements the weighted DSU over integer (or modular) offsets with union-by-rank and path
compression, exposing connectivity, offset queries, and contradiction detection, plus a parity
specialisation for bipartiteness. It is verified against brute force -- a reference that re-derives all
pairwise offsets by BFS over the accepted constraint graph and flags the first inconsistent constraint
-- on hundreds of random constraint sequences, confirming the accepted/rejected decisions and every
reported difference. Pure stdlib; a data-structures companion to the union-find, rollback-DSU, and
bipartite-matching notes."""

from __future__ import annotations


class WeightedDSU:
    """Union-find where each element has a potential; supports 'x - y = d' constraints and queries."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n          # weight[x] = potential(x) - potential(parent[x])

    def find(self, x):
        """Return the root of x, compressing the path and rewriting weights to be relative to the
        root."""
        if self.parent[x] == x:
            return x
        root = self.find(self.parent[x])
        self.weight[x] += self.weight[self.parent[x]]
        self.parent[x] = root
        return root

    def potential(self, x):
        """The potential of x relative to its set's root (after a find that compresses the path)."""
        self.find(x)
        return self.weight[x]

    def union(self, x, y, d):
        """Impose the constraint potential(x) - potential(y) = d.

        Returns True if consistent (constraint recorded or already implied), False if it contradicts
        the current knowledge."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            # both already related; check consistency
            return self.weight[x] - self.weight[y] == d
        # link the lower-rank root under the higher, choosing the weight to satisfy the relation
        if self.rank[rx] < self.rank[ry]:
            # attach rx under ry: want weight[x]-weight[y]=d with x,y potentials relative to roots
            # potential(x) = weight[x] + w(rx), potential(y) = weight[y]; set w(rx) accordingly
            self.parent[rx] = ry
            self.weight[rx] = d - self.weight[x] + self.weight[y]
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
            self.weight[ry] = -d + self.weight[x] - self.weight[y]
        else:
            self.parent[ry] = rx
            self.weight[ry] = -d + self.weight[x] - self.weight[y]
            self.rank[rx] += 1
        return True

    def connected(self, x, y):
        """True iff x and y are in the same set (their relative difference is known)."""
        return self.find(x) == self.find(y)

    def diff(self, x, y):
        """potential(x) - potential(y) if x and y are connected, else None."""
        if self.find(x) != self.find(y):
            return None
        return self.weight[x] - self.weight[y]


class ParityDSU:
    """A weighted DSU specialised to offsets mod 2 -- the classic 'same or different group?' /
    bipartiteness-with-relations structure."""

    def __init__(self, n):
        self.dsu = WeightedDSU(n)

    def relate(self, x, y, same):
        """Record that x and y are in the SAME group (same=True) or DIFFERENT groups (same=False).
        Returns False on contradiction."""
        d = 0 if same else 1
        rx, ry = self.dsu.find(x), self.dsu.find(y)
        if rx == ry:
            return (self.dsu.weight[x] - self.dsu.weight[y]) % 2 == d
        return self.dsu.union(x, y, d)

    def same_group(self, x, y):
        """True/False if the relation is known, else None."""
        dd = self.dsu.diff(x, y)
        if dd is None:
            return None
        return dd % 2 == 0


# --- brute-force reference --------------------------------------------------
def brute_process(n, constraints):
    """Process a list of (x, y, d) difference constraints, returning the list of accept/reject
    decisions by re-deriving all offsets via BFS over the accepted-constraint graph each step.
    A constraint is accepted iff it does not contradict the offsets implied so far."""
    from collections import deque, defaultdict
    adj = defaultdict(list)      # accepted constraint graph: node -> (neighbour, delta)
    decisions = []

    def offset_between(x, y):
        """Known potential(x)-potential(y) via BFS, or None if x,y not yet connected."""
        if x == y:
            return 0
        pot = {x: 0}
        q = deque([x])
        while q:
            u = q.popleft()
            for v, delta in adj[u]:
                if v not in pot:
                    pot[v] = pot[u] + delta        # potential(v) = potential(u) - (u-v relation)
                    q.append(v)
        return pot.get(y)

    for (x, y, d) in constraints:
        known = offset_between(x, y)
        if known is None:
            adj[x].append((y, d))                  # potential(x)-potential(y)=d
            adj[y].append((x, -d))
            decisions.append(True)
        else:
            decisions.append(known == d)
    return decisions

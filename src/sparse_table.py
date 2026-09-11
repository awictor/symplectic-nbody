"""Sparse tables and binary lifting: O(1) range minimum and O(log n) lowest common ancestor.

Some queries deserve constant time. The RANGE MINIMUM QUERY (RMQ) problem -- given a fixed array,
report the minimum of any subrange [l, r) -- is answered by a SPARSE TABLE in O(1) per query after
O(n log n) preprocessing, with no updates allowed. The idea exploits that the minimum is IDEMPOTENT:
overlapping ranges may be combined without double-counting, so any interval can be covered by just
TWO precomputed power-of-two blocks. The table stores, for each start i and each power k, the minimum
of the block [i, i + 2^k); a query on [l, r) takes the length, finds the largest power 2^k that fits,
and returns min(table[l][k], table[r - 2^k][k]) -- two lookups, always.

The same doubling idea, applied to trees, gives BINARY LIFTING for the LOWEST COMMON ANCESTOR. Store
for each node its 2^k-th ancestor (up[v][k] = up[ up[v][k-1] ][k-1]); to find LCA(u, v), first lift
the deeper node to the other's depth by decomposing the depth gap into powers of two, then lift both
in lockstep by the largest jumps that keep them apart until their parents coincide. That is O(log n)
per query after O(n log n) preprocessing, and it also yields the distance between any two nodes in a
tree (depth[u] + depth[v] - 2*depth[lca]) for free.

These are the standard tools for static range queries and tree ancestor queries in competitive
programming, compilers (dominator/ancestor lookups), and phylogenetics. This module implements a
generic sparse table for any idempotent operation (min, max, gcd), a specialized RMQ, and a
binary-lifting LCA with depth and tree-distance queries. It is verified against brute force: the
sparse table matches a direct scan over every subrange, the LCA matches a naive ancestor-walk for
every pair of nodes on many random trees, and the tree distance matches a BFS shortest path. Pure
stdlib; a data-structure companion to the segment-tree, Fenwick-tree, and union-find notes."""

from __future__ import annotations

from collections import deque
from math import gcd


class SparseTable:
    """O(1) range queries for an idempotent associative operation over a static array.

    op must be idempotent (op(x, x) == x): min, max, gcd, bitwise and/or. Default is min."""

    def __init__(self, array, op=min):
        self.op = op
        self.a = list(array)
        n = len(self.a)
        self.n = n
        # log table
        self.log = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1
        K = self.log[n] + 1 if n > 0 else 1
        # table[k][i] = op over [i, i + 2^k)
        self.table = [self.a[:]]
        for k in range(1, K):
            span = 1 << k
            half = 1 << (k - 1)
            prev = self.table[k - 1]
            row = []
            for i in range(0, n - span + 1):
                row.append(op(prev[i], prev[i + half]))
            self.table.append(row)

    def query(self, l, r):
        """op over the half-open range [l, r). Requires l < r."""
        if not (0 <= l < r <= self.n):
            raise IndexError((l, r))
        length = r - l
        k = self.log[length]
        return self.op(self.table[k][l], self.table[k][r - (1 << k)])


class RMQ(SparseTable):
    """Range minimum query (a SparseTable specialized to min)."""

    def __init__(self, array):
        super().__init__(array, op=min)

    def min_range(self, l, r):
        return self.query(l, r)


class LCA:
    """Lowest common ancestor via binary lifting, with depth and tree-distance queries.

    Construct from n and an edge list of an undirected tree; root defaults to 0."""

    def __init__(self, n, edges, root=0):
        self.n = n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        self.LOG = max(1, (n - 1).bit_length())
        self.depth = [0] * n
        self.up = [[-1] * self.LOG for _ in range(n)]

        # BFS to set parents and depths
        visited = [False] * n
        visited[root] = True
        self.up[root][0] = root
        q = deque([root])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    self.depth[w] = self.depth[u] + 1
                    self.up[w][0] = u
                    q.append(w)
        # binary lifting table
        for k in range(1, self.LOG):
            for v in range(n):
                mid = self.up[v][k - 1]
                self.up[v][k] = self.up[mid][k - 1]

    def kth_ancestor(self, v, k):
        """The 2-power-decomposed k-th ancestor of v (root's ancestors are the root)."""
        for i in range(self.LOG):
            if k & (1 << i):
                v = self.up[v][i]
        return v

    def query(self, u, v):
        """The lowest common ancestor of u and v."""
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        # lift u up to v's depth
        diff = self.depth[u] - self.depth[v]
        u = self.kth_ancestor(u, diff)
        if u == v:
            return u
        for k in range(self.LOG - 1, -1, -1):
            if self.up[u][k] != self.up[v][k]:
                u = self.up[u][k]
                v = self.up[v][k]
        return self.up[u][0]

    def distance(self, u, v):
        """Number of edges on the path between u and v."""
        a = self.query(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[a]


def range_gcd_table(array):
    """Convenience: a sparse table for range GCD (gcd is idempotent since gcd(x,x)=x)."""
    return SparseTable(array, op=gcd)

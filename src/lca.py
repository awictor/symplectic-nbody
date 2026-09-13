"""Lowest common ancestor by binary lifting: O(log n) ancestor and LCA queries after O(n log n) prep.

In a rooted tree the LOWEST COMMON ANCESTOR of two nodes u and v is the deepest node that is an
ancestor of both -- the point where the paths up from u and v first meet. It is the workhorse behind
tree distances (dist(u,v) = depth[u] + depth[v] - 2*depth[lca]), path queries, and a dozen
competitive-programming techniques. The naive query walks both nodes up to the root, O(n) each time.

BINARY LIFTING makes each query O(log n). The trick is to precompute, for every node, its 2^k-th
ancestor for all k -- a table up[k][v] where up[0][v] is the parent and

    up[k][v] = up[k-1][ up[k-1][v] ]

(the 2^k-th ancestor is the 2^(k-1)-th ancestor of the 2^(k-1)-th ancestor). Any ancestor jump of
length j is then made by turning on the bits of j: hop by the powers of two that sum to j. To find
the LCA of u and v:

    1. lift the deeper node up until both are at the same depth (a single j-length jump),
    2. if they coincide, that's the LCA,
    3. otherwise lift BOTH together by the largest powers of two that keep them apart; when no jump
       keeps them apart, their common parent is the LCA.

The same table answers the k-th ancestor of a node directly, and the jump from u a given number of
steps toward v (used in path decompositions). This module builds the table from a parent array or an
edge list with a chosen root, and offers lca, kth_ancestor, distance, is_ancestor, and jump.

Validated against a naive reference on random trees: the binary-lifting LCA matches a brute path-to-
root intersection for every pair, distances match a BFS shortest path (a tree has a unique path, so
graph distance = tree distance), k-th ancestor matches walking parents one at a time, and the LCA
identities (lca(u, root)=root, lca(u,u)=u, symmetry) all hold. Pure stdlib; the ancestor-query
companion to the sparse-table RMQ and centroid decomposition."""

from __future__ import annotations

from collections import deque


class LCA:
    """Binary-lifting LCA over a rooted tree on n nodes 0..n-1."""

    def __init__(self, n, edges=None, parent=None, root=0):
        """Build from an undirected edge list (with a chosen root) OR a parent array
        (parent[root] == -1 or root itself). depth[root] = 0."""
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.root = root
        self.LOG = max(1, (n - 1).bit_length())
        self.depth = [0] * n
        self.up = [[-1] * n for _ in range(self.LOG)]

        if parent is not None:
            par = list(parent)
            par[root] = -1
            self._depths_from_parent(par)
        elif edges is not None:
            par = self._bfs_parents(edges)
        else:
            raise ValueError("provide either edges or parent")

        self.up[0] = par
        for k in range(1, self.LOG):
            upk = self.up[k]
            upkm = self.up[k - 1]
            for v in range(n):
                mid = upkm[v]
                upk[v] = upkm[mid] if mid != -1 else -1

    def _bfs_parents(self, edges):
        adj = [[] for _ in range(self.n)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        par = [-1] * self.n
        seen = [False] * self.n
        seen[self.root] = True
        q = deque([self.root])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    par[w] = u
                    self.depth[w] = self.depth[u] + 1
                    q.append(w)
        return par

    def _depths_from_parent(self, par):
        # compute depth by memoised walk to root
        depth = self.depth
        computed = [False] * self.n
        computed[self.root] = True
        for v in range(self.n):
            if computed[v]:
                continue
            chain = []
            x = v
            while x != -1 and not computed[x]:
                chain.append(x)
                x = par[x]
            base = depth[x] if x != -1 else 0
            for node in reversed(chain):
                base += 1
                depth[node] = base
                computed[node] = True

    def kth_ancestor(self, v, k):
        """The k-th ancestor of v (k steps toward the root), or -1 if it goes past the root."""
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > self.depth[v]:
            return -1
        for i in range(self.LOG):
            if k & (1 << i):
                v = self.up[i][v]
                if v == -1:
                    return -1
        return v

    def lca(self, u, v):
        """Lowest common ancestor of u and v."""
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        # lift u up to v's depth
        u = self.kth_ancestor(u, self.depth[u] - self.depth[v])
        if u == v:
            return u
        for i in range(self.LOG - 1, -1, -1):
            if self.up[i][u] != self.up[i][v]:
                u = self.up[i][u]
                v = self.up[i][v]
        return self.up[0][u]

    def distance(self, u, v):
        """Number of edges on the unique path from u to v."""
        w = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[w]

    def is_ancestor(self, u, v):
        """True if u is an ancestor of v (u == v counts)."""
        return self.lca(u, v) == u

    def jump(self, u, v, k):
        """The node k steps along the path from u toward v, or -1 if k exceeds the path length."""
        w = self.lca(u, v)
        du = self.depth[u] - self.depth[w]  # steps up from u to the LCA
        dist = du + (self.depth[v] - self.depth[w])
        if k < 0 or k > dist:
            return -1
        if k <= du:
            return self.kth_ancestor(u, k)
        # descend toward v: it's the (dist-k)-th ancestor of v
        return self.kth_ancestor(v, dist - k)


# --- naive references --------------------------------------------------------
def _ancestors_to_root(v, parent):
    chain = []
    while v != -1:
        chain.append(v)
        v = parent[v]
    return chain


def naive_lca(u, v, parent):
    """Independent reference: intersect the two root paths."""
    au = set(_ancestors_to_root(u, parent))
    x = v
    while x != -1:
        if x in au:
            return x
        x = parent[x]
    return -1

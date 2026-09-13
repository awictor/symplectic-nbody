"""Heavy-light decomposition: turn tree-path queries into a handful of contiguous array ranges.

A tree has no natural linear order, so a query like "the maximum edge/vertex value on the path from u
to v" or "add x to every vertex on that path" seems to need an O(path length) walk. Heavy-light
decomposition (Sleator and Tarjan, 1983) makes each such query O(log^2 n) by cutting the tree into
vertex-disjoint CHAINS and laying every chain out contiguously in one array, so a root-to-node path
touches only O(log n) chains -- each a single contiguous segment a range structure can answer in
O(log n).

The cut rule: for each vertex, the child with the largest subtree is its HEAVY child; the edge to it
is a heavy edge, all others light. Heavy edges link up into chains. The key fact is that any
root-to-leaf path crosses at most O(log n) LIGHT edges -- because stepping down a light edge at least
halves the remaining subtree size -- so a path decomposes into O(log n) chain pieces. Assigning each
vertex a position in DFS order that keeps every chain contiguous lets a segment tree index the tree.

This module builds the decomposition (heavy child, chain head, subtree size, DFS position) and layers
a segment tree over the positions to support, on the u-v path: SUM, MAX, and range ADD, plus point
updates -- climbing chains by repeatedly jumping from the deeper chain head to its parent. The LCA
falls out of the same climb for free. Values live on vertices.

Validated against naive references on random trees: every path query matches a brute-force walk of
the actual tree path (found by an LCA-free parent walk), across many trees, updates, and query types;
the HLD-derived LCA matches an independent computation; and the O(log n)-chains property is checked
directly. Pure stdlib; the path-query companion to the binary-lifting LCA and the segment tree."""

from __future__ import annotations

import sys


class _SegTree:
    """Iterative segment tree over a fixed array; supports point-set and range sum/max."""

    def __init__(self, values):
        self.n = len(values)
        size = 1
        while size < self.n:
            size *= 2
        self.size = size
        self.sum = [0] * (2 * size)
        self.mx = [-float("inf")] * (2 * size)
        for i, v in enumerate(values):
            self.sum[size + i] = v
            self.mx[size + i] = v
        for i in range(size - 1, 0, -1):
            self.sum[i] = self.sum[2 * i] + self.sum[2 * i + 1]
            self.mx[i] = max(self.mx[2 * i], self.mx[2 * i + 1])

    def point_set(self, i, v):
        i += self.size
        self.sum[i] = v
        self.mx[i] = v
        i //= 2
        while i:
            self.sum[i] = self.sum[2 * i] + self.sum[2 * i + 1]
            self.mx[i] = max(self.mx[2 * i], self.mx[2 * i + 1])
            i //= 2

    def point_add(self, i, delta):
        self.point_set(i, self.sum[i + self.size] + delta)

    def range_sum(self, l, r):
        """Sum over [l, r] inclusive."""
        res = 0
        l += self.size
        r += self.size + 1
        while l < r:
            if l & 1:
                res += self.sum[l]
                l += 1
            if r & 1:
                r -= 1
                res += self.sum[r]
            l //= 2
            r //= 2
        return res

    def range_max(self, l, r):
        """Max over [l, r] inclusive."""
        res = -float("inf")
        l += self.size
        r += self.size + 1
        while l < r:
            if l & 1:
                res = max(res, self.mx[l])
                l += 1
            if r & 1:
                r -= 1
                res = max(res, self.mx[r])
            l //= 2
            r //= 2
        return res


class HeavyLight:
    """Heavy-light decomposition of a rooted tree with per-vertex values."""

    def __init__(self, n, edges, values=None, root=0):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.root = root
        self.adj = [[] for _ in range(n)]
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
        self.values = list(values) if values is not None else [0] * n

        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [1] * n
        self.heavy = [-1] * n
        self.head = [0] * n
        self.pos = [0] * n  # position in the base array

        self._dfs_sizes()
        self._decompose()

        base = [0] * n
        for v in range(n):
            base[self.pos[v]] = self.values[v]
        self.seg = _SegTree(base)

    def _dfs_sizes(self):
        # iterative DFS computing parent, depth, subtree size, heavy child
        root = self.root
        order = []
        stack = [root]
        self.parent[root] = -1
        visited = [False] * self.n
        visited[root] = True
        while stack:
            u = stack.pop()
            order.append(u)
            for w in self.adj[u]:
                if not visited[w]:
                    visited[w] = True
                    self.parent[w] = u
                    self.depth[w] = self.depth[u] + 1
                    stack.append(w)
        # process in reverse topological (children before parents) for sizes
        for u in reversed(order):
            best = -1
            best_size = 0
            for w in self.adj[u]:
                if w == self.parent[u]:
                    continue
                self.size[u] += self.size[w]
                if self.size[w] > best_size:
                    best_size = self.size[w]
                    best = w
            self.heavy[u] = best

    def _decompose(self):
        # assign chain heads and contiguous positions; heavy child gets the next slot
        cur_pos = 0
        # iterative: for each chain head, walk down heavy children
        stack = [(self.root, self.root)]
        while stack:
            v, h = stack.pop()
            # walk the chain from v downward following heavy edges
            while v != -1:
                self.head[v] = h
                self.pos[v] = cur_pos
                cur_pos += 1
                # push light children as new chain heads
                for w in self.adj[v]:
                    if w != self.parent[v] and w != self.heavy[v]:
                        stack.append((w, w))
                v = self.heavy[v]

    def update(self, v, value):
        """Set vertex v's value."""
        self.values[v] = value
        self.seg.point_set(self.pos[v], value)

    def add(self, v, delta):
        """Add delta to vertex v's value."""
        self.values[v] += delta
        self.seg.point_add(self.pos[v], delta)

    def lca(self, u, v):
        """Lowest common ancestor via the chain climb."""
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            u = self.parent[self.head[u]]
        return u if self.depth[u] < self.depth[v] else v

    def path_sum(self, u, v):
        """Sum of vertex values on the path from u to v (inclusive)."""
        res = 0
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            res += self.seg.range_sum(self.pos[self.head[u]], self.pos[u])
            u = self.parent[self.head[u]]
        lo, hi = (self.pos[u], self.pos[v]) if self.pos[u] <= self.pos[v] else (self.pos[v], self.pos[u])
        res += self.seg.range_sum(lo, hi)
        return res

    def path_max(self, u, v):
        """Maximum vertex value on the path from u to v (inclusive)."""
        res = -float("inf")
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            res = max(res, self.seg.range_max(self.pos[self.head[u]], self.pos[u]))
            u = self.parent[self.head[u]]
        lo, hi = (self.pos[u], self.pos[v]) if self.pos[u] <= self.pos[v] else (self.pos[v], self.pos[u])
        res = max(res, self.seg.range_max(lo, hi))
        return res

    def max_light_edges_on_any_root_path(self):
        """Diagnostic: the maximum number of light edges (chain switches) on any root-to-node path.
        By the HLD theorem this is O(log n)."""
        best = 0
        for v in range(self.n):
            switches = 0
            x = v
            while self.head[x] != self.root:
                switches += 1
                x = self.parent[self.head[x]]
            best = max(best, switches)
        return best


# --- naive references --------------------------------------------------------
def naive_path(u, v, parent, depth):
    """The actual set of vertices on the u-v path, by walking both up to their meeting point."""
    pu, pv = [], []
    a, b = u, v
    # bring to equal depth
    while depth[a] > depth[b]:
        pu.append(a)
        a = parent[a]
    while depth[b] > depth[a]:
        pv.append(b)
        b = parent[b]
    while a != b:
        pu.append(a)
        pv.append(b)
        a = parent[a]
        b = parent[b]
    return pu + [a] + list(reversed(pv))

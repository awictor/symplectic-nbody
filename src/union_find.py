"""Union-Find (disjoint-set union): tracking connectivity in near-constant time.

Given n items and a stream of "these two are connected" facts, answer at any moment: are these
two in the same group? how many groups are there? This is the disjoint-set problem, and the
Union-Find structure solves it so efficiently that a sequence of m operations on n items runs in
O(m * alpha(n)) time -- where alpha is the inverse Ackermann function, which is at most 4 for
any number of items that could exist in the universe. Effectively constant per operation.

Two ideas make it fast. Each set is a tree whose root is its representative; find(x) walks to
the root. UNION BY RANK attaches the shorter tree under the taller so trees stay shallow. PATH
COMPRESSION flattens the path on the way up -- every node visited during a find is repointed
straight at the root, so repeated queries get faster. Together they give the near-constant bound
that Tarjan proved tight.

Union-Find is the engine behind Kruskal's minimum-spanning-tree algorithm (add the cheapest edge
that does not form a cycle -- a cycle is exactly two endpoints already in the same set), image
segmentation, network-connectivity and percolation, and the "friend circles" and account-merging
problems. This module implements find with path compression, union by rank, connected-component
counting, and a Kruskal MST built on top, and checks the components against a brute-force
flood-fill. Pure stdlib; the graph-structure companion to the Fenwick-tree note.
"""

from __future__ import annotations


class UnionFind:
    """Disjoint-set union over n elements 0..n-1 with union by rank and path compression."""

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("size must be nonnegative")
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n          # size of the set rooted at each root
        self.num_sets = n            # current number of disjoint sets

    def find(self, x: int) -> int:
        """Return the representative (root) of x's set, compressing the path along the way."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression: repoint every node on the path directly at the root
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        """Merge the sets containing a and b. Returns True if they were separate (a merge
        happened), False if they were already connected."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # union by rank: hang the shorter tree under the taller
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.num_sets -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        """True if a and b are in the same set."""
        return self.find(a) == self.find(b)

    def set_size(self, x: int) -> int:
        """Size of the set containing x."""
        return self.size[self.find(x)]

    def count(self) -> int:
        """Number of disjoint sets remaining."""
        return self.num_sets

    def components(self):
        """Return the sets as a list of sorted member lists."""
        groups = {}
        for x in range(len(self.parent)):
            groups.setdefault(self.find(x), []).append(x)
        return [sorted(v) for v in groups.values()]


def connected_components(n: int, edges):
    """Number of connected components in an undirected graph of n nodes and the given edge list
    (pairs of node indices), via Union-Find."""
    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)
    return uf.count()


def kruskal_mst(n: int, weighted_edges):
    """Minimum spanning tree (or forest) by Kruskal's algorithm: sort edges by weight and add
    each that joins two different components. `weighted_edges` is a list of (weight, u, v).
    Returns (total_weight, chosen_edges)."""
    uf = UnionFind(n)
    total = 0
    chosen = []
    for w, u, v in sorted(weighted_edges):
        if uf.union(u, v):          # adds the edge only if it connects two separate trees
            total += w
            chosen.append((w, u, v))
            if len(chosen) == n - 1:  # a spanning tree is complete
                break
    return total, chosen

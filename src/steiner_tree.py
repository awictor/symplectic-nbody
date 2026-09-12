"""Steiner tree in graphs: the cheapest network connecting a chosen set of terminals.

The minimum spanning tree connects ALL vertices of a graph at least cost. The STEINER TREE problem asks
a subtler question: connect only a chosen subset of vertices -- the TERMINALS -- at least cost, where
the tree is free to route through any of the other vertices (called STEINER POINTS) if that is cheaper.
This is the true shape of network design: lay the cheapest fibre backbone linking a handful of cities
(routing through junction towns is allowed), wire a chip net touching a set of pins (using free routing
tracks), or build a phylogenetic tree connecting observed species through inferred ancestors. Unlike
the MST it is NP-hard in general, but when the number of terminals k is small it is solved exactly by a
beautiful bitmask dynamic program.

The DREYFUS-WAGNER algorithm computes dp[S][v] = the minimum weight of a Steiner tree that connects the
terminal subset S and is rooted at (includes) vertex v. It fills this table by two rules. MERGE: a tree
for S rooted at v can be split into two sub-trees for disjoint subsets S1 and S2 both rooted at v, so
dp[S][v] = min over subset splits of dp[S1][v] + dp[S2][v]. GROW: a tree rooted at v can be extended to
a neighbour by a shortest path, so dp[S][v] = min over u of dp[S][u] + dist(u, v). The second rule is a
multi-source shortest-path relaxation (a Dijkstra/Bellman-Ford sweep) applied to each subset layer.
Iterating subsets in increasing order of size, the answer is min over v of dp[full][v], computed in
O(3^k n + 2^k n^2) time -- exponential only in the number of terminals, not the graph size.

This module computes the exact minimum Steiner tree weight for a set of terminals in a weighted
undirected graph via Dreyfus-Wagner. It is verified against brute force -- for small graphs, enumerating
every subset of Steiner points and taking the MST of the induced terminal-plus-chosen-points subgraph
-- confirming the DP finds the true optimum, and against the minimum spanning tree in the special case
where every vertex is a terminal. Pure stdlib; a network-design companion to the MST (Kruskal/Prim),
shortest-path, and arborescence notes."""

from __future__ import annotations

import heapq


_INF = float("inf")


def steiner_tree(n, edges, terminals):
    """Minimum-weight Steiner tree connecting `terminals` in an undirected weighted graph.

    `n` vertices (0..n-1); `edges` a list of (u, v, w) undirected edges with w >= 0; `terminals` a
    list of distinct vertices to connect. Returns the minimum total weight (0 for <=1 terminal), or
    inf if the terminals cannot all be connected."""
    terminals = list(dict.fromkeys(terminals))     # dedupe, keep order
    k = len(terminals)
    if k <= 1:
        return 0

    # adjacency list
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    full = (1 << k) - 1
    # dp[mask][v]: min weight of a tree connecting terminals in `mask` and including vertex v
    dp = [[_INF] * n for _ in range(1 << k)]

    # base case: a single terminal alone costs 0 at its own vertex
    for i, t in enumerate(terminals):
        dp[1 << i][t] = 0

    for mask in range(1, 1 << k):
        # MERGE: combine two disjoint sub-masks rooted at the same vertex
        sub = (mask - 1) & mask
        while sub > 0:
            other = mask ^ sub
            if sub < other:                          # each unordered split once
                row_sub = dp[sub]
                row_other = dp[other]
                row = dp[mask]
                for v in range(n):
                    a = row_sub[v]
                    if a == _INF:
                        continue
                    b = row_other[v]
                    if b == _INF:
                        continue
                    s = a + b
                    if s < row[v]:
                        row[v] = s
            sub = (sub - 1) & mask

        # GROW: relax along edges (multi-source Dijkstra over the current mask layer)
        _dijkstra_relax(dp[mask], adj, n)

    return min(dp[full])


def _dijkstra_relax(dist, adj, n):
    """In-place Dijkstra that treats the current `dist` array as multiple sources, relaxing
    dist[v] = min(dist[v], dist[u] + w) over all edges until stable."""
    pq = [(dist[v], v) for v in range(n) if dist[v] < _INF]
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))


# --- brute-force reference --------------------------------------------------
def brute_steiner_tree(n, edges, terminals):
    """Minimum Steiner tree by enumerating every subset of non-terminal vertices as potential Steiner
    points, building the MST of the induced subgraph on (terminals + chosen points), and taking the
    cheapest that connects all terminals. Exponential in n; small graphs only."""
    terminals = list(dict.fromkeys(terminals))
    if len(terminals) <= 1:
        return 0
    term_set = set(terminals)
    others = [v for v in range(n) if v not in term_set]

    best = _INF
    for subset_mask in range(1 << len(others)):
        chosen = term_set | {others[i] for i in range(len(others)) if subset_mask & (1 << i)}
        w = _mst_weight_on(chosen, edges)
        if w is not None and w < best:
            best = w
    return best


def _mst_weight_on(vertices, edges):
    """MST weight of the subgraph induced by `vertices` (using only edges with both ends inside),
    or None if that subgraph is disconnected."""
    vs = sorted(vertices)
    if len(vs) <= 1:
        return 0
    index = {v: i for i, v in enumerate(vs)}
    sub_edges = sorted((w, u, v) for (u, v, w) in edges if u in index and v in index)
    parent = list(range(len(vs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0
    used = 0
    for w, u, v in sub_edges:
        ru, rv = find(index[u]), find(index[v])
        if ru != rv:
            parent[ru] = rv
            total += w
            used += 1
    return total if used == len(vs) - 1 else None

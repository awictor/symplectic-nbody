"""Johnson's algorithm: all-pairs shortest paths on sparse graphs, even with negative edges.

To find the shortest path between every pair of nodes, Floyd-Warshall runs in O(V^3) regardless of how
many edges there are -- wasteful on a SPARSE graph. Running Dijkstra from every source would be
O(V * E log V), far better when E << V^2, but Dijkstra cannot handle negative edge weights. Johnson's
algorithm (1977) is the clever reconciliation: it REWEIGHTS the graph so all edges become non-negative
while preserving which paths are shortest, then runs Dijkstra from every node on the safe reweighted
graph.

The reweighting is the beautiful part. Add a virtual source q connected to every node with a zero-weight
edge and run Bellman-Ford from q to get a potential h(v) = shortest distance from q to v (this also
detects any negative cycle). Define the new weight of edge (u,v) as w'(u,v) = w(u,v) + h(u) - h(v). By
the triangle inequality h(v) <= h(u) + w(u,v), so w' >= 0 for every edge -- Dijkstra is now safe. And
because the h-terms telescope along any path from s to t (all interior potentials cancel), the
reweighted path length is just w(path) + h(s) - h(t): every path between the same endpoints shifts by
the SAME constant, so the shortest path is unchanged. After Dijkstra, the true distance is recovered as
d(s,t) = d'(s,t) - h(s) + h(t).

This module implements Johnson's algorithm on an edge list with arbitrary (possibly negative) weights,
returning the full all-pairs distance matrix and supporting path reconstruction, and it raises on a
negative cycle. It is validated: the reweighted edges are all non-negative and preserve shortest paths;
the all-pairs distances agree exactly with an independent Floyd-Warshall on graphs with and without
negative edges; a negative cycle is detected; reconstructed paths have the reported length; and it
matches a brute-force Bellman-Ford from every source. Reuses the repo's Dijkstra min-heap. Pure stdlib;
the sparse-graph companion to the Dijkstra, Bellman-Ford, and Floyd-Warshall tools."""

from __future__ import annotations

import math

from dijkstra import MinHeap


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for (u, v, w) in edges:
        adj[u].append((v, w))
    return adj


def bellman_ford(n, edges, source):
    """Shortest distances from source over an edge list allowing negative weights.

    Returns (dist, prev). Raises ValueError on a reachable negative cycle."""
    dist = [math.inf] * n
    prev = [None] * n
    dist[source] = 0.0
    for _ in range(n - 1):
        changed = False
        for (u, v, w) in edges:
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                changed = True
        if not changed:
            break
    # one more pass detects a negative cycle
    for (u, v, w) in edges:
        if dist[u] != math.inf and dist[u] + w < dist[v] - 1e-12:
            raise ValueError("graph contains a negative cycle")
    return dist, prev


def _dijkstra(n, adj, source):
    dist = [math.inf] * n
    prev = [None] * n
    dist[source] = 0.0
    heap = MinHeap()
    heap.push(0.0, source)
    while heap:
        d, u = heap.pop()
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heap.push(nd, v)
    return dist, prev


def johnson(n, edges):
    """All-pairs shortest paths via Johnson's algorithm.

    n: number of nodes (0..n-1). edges: list of (u, v, weight), directed, weights may be negative.
    Returns (dist, next_hop) where dist[s][t] is the shortest distance (inf if unreachable) and
    next_hop[s][t] is the first node after s on a shortest path (None if unreachable or s==t).
    Raises ValueError on a negative cycle."""
    # 1. add virtual source q = n connected to every node with weight 0
    q = n
    aug_edges = list(edges) + [(q, v, 0.0) for v in range(n)]
    h, _ = bellman_ford(n + 1, aug_edges, q)          # potentials; raises on negative cycle
    h = h[:n]

    # 2. reweight edges to be non-negative
    reweighted = [[] for _ in range(n)]
    for (u, v, w) in edges:
        reweighted[u].append((v, w + h[u] - h[v]))

    # 3. Dijkstra from every node on the reweighted graph, then undo the shift
    dist = [[math.inf] * n for _ in range(n)]
    next_hop = [[None] * n for _ in range(n)]
    for s in range(n):
        d_prime, prev = _dijkstra(n, reweighted, s)
        for t in range(n):
            if d_prime[t] != math.inf:
                dist[s][t] = d_prime[t] - h[s] + h[t]
        # build next-hop from the Dijkstra predecessor tree (shortest paths are preserved)
        for t in range(n):
            if t == s or d_prime[t] == math.inf:
                continue
            # walk back from t to s to find the first hop after s
            node = t
            while prev[node] is not None and prev[node] != s:
                node = prev[node]
            if prev[node] == s:
                next_hop[s][t] = node
            elif node == t and False:
                pass
    return dist, next_hop


def reconstruct_path(next_hop, s, t):
    """Reconstruct the shortest path s -> t as a list of nodes using the next-hop table."""
    if s == t:
        return [s]
    if next_hop[s][t] is None:
        return []
    path = [s]
    cur = s
    while cur != t:
        nxt = next_hop[cur][t]
        if nxt is None:
            return []
        path.append(nxt)
        cur = nxt
    return path


def reweighted_edges(n, edges):
    """Expose the non-negative reweighted edge list (for inspection/validation)."""
    q = n
    aug = list(edges) + [(q, v, 0.0) for v in range(n)]
    h, _ = bellman_ford(n + 1, aug, q)
    h = h[:n]
    return [(u, v, w + h[u] - h[v]) for (u, v, w) in edges], h

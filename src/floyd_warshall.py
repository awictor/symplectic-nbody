"""Floyd-Warshall: shortest paths between every pair of nodes.

Dijkstra gives shortest paths from ONE source; sometimes you need them between ALL pairs -- for
a routing table, a road-network distance matrix, or the "degrees of separation" in a graph.
Running Dijkstra from every node works, but Floyd-Warshall (1962) does it in one elegant O(V^3)
dynamic program that also handles NEGATIVE edge weights (which Dijkstra cannot) and detects
negative cycles.

The idea is to allow ever-larger sets of intermediate nodes. Let dist[i][j] be the shortest path
from i to j using only nodes {0, 1, ..., k} as intermediates. Adding node k to the allowed set,

    dist[i][j] = min( dist[i][j],                 don't route through k
                      dist[i][k] + dist[k][j] )   do route through k,

and after k runs over every node, dist[i][j] is the true shortest distance. Three nested loops,
no priority queue. Recording, for each pair, the next hop on the shortest path lets you
reconstruct the actual route. A negative value on the diagonal after the algorithm finishes means
node i lies on a NEGATIVE CYCLE -- a loop you can traverse to lower the cost without bound, so no
shortest path is defined.

Dropping the weights and using boolean OR instead of min gives the TRANSITIVE CLOSURE: which
nodes are reachable from which. This module computes the all-pairs distance matrix, reconstructs
paths, detects negative cycles, and builds the transitive closure, and checks the distances
against running Dijkstra from every source. Pure stdlib; the all-pairs companion to the Dijkstra
note.
"""

from __future__ import annotations

import math

INF = math.inf


def floyd_warshall(n: int, edges):
    """All-pairs shortest paths for an n-node graph. `edges` is a list of (u, v, weight) directed
    edges (weights may be negative). Returns (dist, nxt) where dist[i][j] is the shortest
    distance (INF if unreachable) and nxt[i][j] is the next node after i on a shortest path to j
    (None if no path). Raises ValueError if a negative cycle is present."""
    dist = [[INF] * n for _ in range(n)]
    nxt = [[None] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        nxt[i][i] = i
    for u, v, w in edges:
        if w < dist[u][v]:          # keep the cheapest parallel edge
            dist[u][v] = w
            nxt[u][v] = v
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue            # i can't reach k; no improvement via k
            di = dist[i]
            ni = nxt[i]
            for j in range(n):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd
                    ni[j] = nxt[i][k]
    for i in range(n):
        if dist[i][i] < 0:
            raise ValueError("graph contains a negative cycle")
    return dist, nxt


def reconstruct_path(nxt, u, v):
    """The shortest path from u to v as a list of nodes, using the `nxt` matrix. Empty if v is
    unreachable from u."""
    if nxt[u][v] is None:
        return []
    path = [u]
    while u != v:
        u = nxt[u][v]
        path.append(u)
    return path


def has_negative_cycle(n: int, edges) -> bool:
    """True if the graph contains a negative cycle (run Floyd-Warshall and check the diagonal)."""
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w
    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return any(dist[i][i] < 0 for i in range(n))


def transitive_closure(n: int, edges):
    """Reachability matrix: reach[i][j] is True if j is reachable from i (Floyd-Warshall with
    boolean OR instead of min). Every node reaches itself."""
    reach = [[False] * n for _ in range(n)]
    for i in range(n):
        reach[i][i] = True
    for u, v, _w in _as_weighted(edges):
        reach[u][v] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                rk = reach[k]
                ri = reach[i]
                for j in range(n):
                    if rk[j]:
                        ri[j] = True
    return reach


def _as_weighted(edges):
    """Accept edges as (u, v) or (u, v, w); yield (u, v, w) with w defaulting to 1."""
    for e in edges:
        if len(e) == 2:
            yield e[0], e[1], 1
        else:
            yield e[0], e[1], e[2]


# --- Dijkstra reference for validation (nonnegative weights) ---------------

def dijkstra_all_pairs(n: int, edges):
    """All-pairs distances by running a simple Dijkstra from every source -- used to check
    Floyd-Warshall on nonnegative-weight graphs."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [[INF] * n for _ in range(n)]
    for s in range(n):
        d = dist[s]
        d[s] = 0
        visited = [False] * n
        for _ in range(n):
            # pick the unvisited node with the smallest tentative distance
            u, best = -1, INF
            for x in range(n):
                if not visited[x] and d[x] < best:
                    best, u = d[x], x
            if u == -1:
                break
            visited[u] = True
            for v, w in adj[u]:
                if d[u] + w < d[v]:
                    d[v] = d[u] + w
    return dist

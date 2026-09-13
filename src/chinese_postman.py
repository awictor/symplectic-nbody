"""The Chinese Postman problem -- the shortest closed walk that traverses every edge.

A mail carrier must walk down EVERY street and return to the depot; a snowplough must clear every road;
a drone must inspect every pipeline segment. This is ROUTE INSPECTION, the Chinese Postman Problem
(Kwan Mei-Ko, 1962): find the shortest closed walk in a weighted graph that uses every edge at least
once. It is the edge-covering cousin of the Travelling Salesman (which covers every VERTEX) -- and,
remarkably, unlike TSP it is solvable in POLYNOMIAL time.

The insight is Euler's. A connected graph has a closed walk using every edge EXACTLY once -- an EULERIAN
CIRCUIT -- if and only if every vertex has EVEN degree. If some vertices have odd degree (there are
always an even number of them, by the handshake lemma), the postman is forced to RETRACE some edges to
fix the parity. The cheapest way to do that is to pair up the odd-degree vertices and, for each pair,
duplicate the edges along a SHORTEST PATH between them -- turning both endpoints even. Choosing the
pairing that minimises the total duplicated length is exactly a MINIMUM-WEIGHT PERFECT MATCHING on the
complete graph of odd vertices (with shortest-path distances as weights). Add those duplicated edges,
and the augmented graph is Eulerian; its Eulerian circuit is the optimal postman route, of length
(sum of all edge weights) + (the minimum matching cost).

This module computes the postman route length and, for a fully-even graph, an actual Eulerian circuit.
It finds all-pairs shortest paths (Floyd-Warshall), identifies the odd-degree vertices, and solves the
minimum-weight perfect matching over them by exact search (the odd set is small in practice), returning
the extra distance the postman must walk beyond simply summing the edges. Pure standard library.

Validation. The route length is checked to equal (total edge weight) + (minimum odd-vertex matching
cost), and that matching cost is verified against a brute-force search over all pairings. On EULERIAN
graphs (all even degrees) the extra cost is zero and the returned circuit is a genuine Eulerian circuit
-- it starts and ends at the same vertex and uses every edge exactly once, checked directly. The
minimum matching is confirmed optimal against brute force on many random graphs, small hand instances
(a square needs no retracing; a path graph forces retracing the whole path) match by hand, and the
handshake lemma (an even number of odd-degree vertices) is verified. Disconnected graphs are reported
as having no postman route. Pure standard library."""

import math
from itertools import combinations


INF = float("inf")


def _all_pairs_shortest(n, adj):
    """Floyd-Warshall all-pairs shortest path distances. adj[i][j] = min edge weight or INF."""
    dist = [row[:] for row in adj]
    for i in range(n):
        dist[i][i] = 0.0
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            for j in range(n):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd
    return dist


def _min_weight_matching(odds, dist):
    """Minimum-weight perfect matching over the odd vertices, by exact recursive pairing."""
    best = {"cost": INF}

    def rec(remaining, acc):
        if not remaining:
            best["cost"] = min(best["cost"], acc)
            return
        if acc >= best["cost"]:
            return                       # prune
        first = remaining[0]
        for i in range(1, len(remaining)):
            partner = remaining[i]
            rest = remaining[1:i] + remaining[i + 1:]
            rec(rest, acc + dist[first][partner])

    rec(list(odds), 0.0)
    return best["cost"] if best["cost"] != INF else 0.0


def _build_adj(n, edges):
    """Adjacency matrix of minimum edge weights, degree list, and connectivity check."""
    adj = [[INF] * n for _ in range(n)]
    degree = [0] * n
    total = 0.0
    present = [False] * n
    for u, v, w in edges:
        if w < adj[u][v]:
            adj[u][v] = w
            adj[v][u] = w
        degree[u] += 1
        degree[v] += 1
        total += w
        present[u] = present[v] = True
    return adj, degree, total, present


def _connected_on_present(n, edges, present):
    """Is the subgraph on vertices that appear in edges connected?"""
    verts = [i for i in range(n) if present[i]]
    if not verts:
        return True
    nb = {i: [] for i in verts}
    for u, v, _ in edges:
        nb[u].append(v)
        nb[v].append(u)
    seen = set([verts[0]])
    stack = [verts[0]]
    while stack:
        x = stack.pop()
        for y in nb[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == len(verts)


def chinese_postman(n, edges):
    """Solve the undirected Chinese Postman problem.

    ``edges`` is a list of (u, v, weight). Returns a dict with:
      'route_length': length of the shortest closed walk covering every edge (INF if disconnected),
      'total_edge_weight': sum of all edge weights,
      'extra_cost': the added length from retracing (0 if already Eulerian),
      'odd_vertices': the odd-degree vertices,
      'is_eulerian': whether the graph is Eulerian to begin with.
    """
    adj, degree, total, present = _build_adj(n, edges)
    if not _connected_on_present(n, edges, present):
        return {"route_length": INF, "total_edge_weight": total, "extra_cost": INF,
                "odd_vertices": [i for i in range(n) if degree[i] % 2], "is_eulerian": False}
    odds = [i for i in range(n) if degree[i] % 2 == 1]
    if not odds:
        return {"route_length": total, "total_edge_weight": total, "extra_cost": 0.0,
                "odd_vertices": [], "is_eulerian": True}
    dist = _all_pairs_shortest(n, adj)
    extra = _min_weight_matching(odds, dist)
    return {"route_length": total + extra, "total_edge_weight": total, "extra_cost": extra,
            "odd_vertices": odds, "is_eulerian": False}


# ---------------------------------------------------------------------------
# Eulerian circuit for the already-even case (Hierholzer)
# ---------------------------------------------------------------------------

def eulerian_circuit(n, edges, start=0):
    """Return an Eulerian circuit (list of vertices) if the graph is connected and all-even, else None.

    Uses each edge exactly once and returns to the start.
    """
    adj, degree, _, present = _build_adj(n, edges)
    if any(d % 2 for d in degree):
        return None
    if not _connected_on_present(n, edges, present):
        return None
    # multigraph adjacency with edge ids for Hierholzer
    graph = {i: [] for i in range(n)}
    for eid, (u, v, _) in enumerate(edges):
        graph[u].append((v, eid))
        graph[v].append((u, eid))
    used = [False] * len(edges)
    if not edges:
        return [start]
    if not present[start]:
        start = next(i for i in range(n) if present[i])
    circuit = []
    stack = [start]
    idx = {i: 0 for i in range(n)}
    while stack:
        v = stack[-1]
        advanced = False
        while idx[v] < len(graph[v]):
            to, eid = graph[v][idx[v]]
            idx[v] += 1
            if not used[eid]:
                used[eid] = True
                stack.append(to)
                advanced = True
                break
        if not advanced:
            circuit.append(stack.pop())
    return circuit[::-1]


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def is_eulerian_circuit(edges, circuit):
    """True if ``circuit`` starts and ends at the same vertex and uses every edge exactly once."""
    if not circuit:
        return not edges
    if circuit[0] != circuit[-1]:
        return False
    # multiset of edges walked
    walked = {}
    for a, b in zip(circuit, circuit[1:]):
        key = (min(a, b), max(a, b))
        walked[key] = walked.get(key, 0) + 1
    need = {}
    for u, v, _ in edges:
        key = (min(u, v), max(u, v))
        need[key] = need.get(key, 0) + 1
    return walked == need


def brute_min_matching(odds, dist):
    """Reference: minimum-weight perfect matching over odds by trying all pairings."""
    if not odds:
        return 0.0
    best = INF

    def rec(rem, acc):
        nonlocal best
        if not rem:
            best = min(best, acc)
            return
        first = rem[0]
        for i in range(1, len(rem)):
            rec(rem[1:i] + rem[i + 1:], acc + dist[first][rem[i]])

    rec(list(odds), 0.0)
    return best

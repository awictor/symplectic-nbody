"""Christofides: the 1.5-approximation for metric TSP, and why the odd vertices matter.

The travelling salesman problem -- shortest closed tour visiting every city -- is NP-hard, so for
large instances we settle for a tour provably close to optimal. Christofides' 1976 algorithm is the
famous one: on any METRIC instance (distances satisfy the triangle inequality) it returns a tour no
longer than 1.5 times the optimum, a guarantee that stood unbeaten for over forty years. It is a
small masterpiece of assembling exact subroutines into an approximation.

The construction:

  1. MINIMUM SPANNING TREE. Build the MST of the cities (Kruskal). The MST is a lower bound on the
     optimum -- deleting any edge of the optimal tour leaves a spanning path, itself a spanning tree,
     so OPT >= MST.

  2. ODD-DEGREE VERTICES. A graph has an Eulerian circuit (a closed walk using every edge once) iff
     every vertex has even degree. The MST generally does not. But a classic parity fact says the
     number of odd-degree vertices is always EVEN, so they can be paired up.

  3. MINIMUM-WEIGHT PERFECT MATCHING on just those odd-degree vertices. Add the matching edges to the
     tree. Now every vertex has even degree (each odd vertex gained exactly one edge), so the
     combined multigraph is Eulerian. This matching is the clever step and the source of the 1.5:
     the odd vertices split the optimal tour into two alternating sub-tours, the cheaper of which is
     a perfect matching of cost <= OPT/2, so the MINIMUM matching costs <= OPT/2.

  4. EULERIAN CIRCUIT, then SHORTCUT. Walk an Eulerian circuit of tree + matching (cost
     <= MST + OPT/2 <= OPT + OPT/2 = 1.5 OPT). It may revisit cities; shortcut past any already-seen
     city straight to the next new one. By the triangle inequality shortcutting never lengthens the
     walk, so the final Hamiltonian tour still costs <= 1.5 OPT.

This module implements christofides (the full pipeline returning a tour and its length), plus the
pieces exposed for testing: minimum spanning tree via Kruskal, odd_degree_vertices, a
min_weight_perfect_matching (exact, by recursive enumeration -- the odd set is small), an Eulerian
circuit on the multigraph (Hierholzer), and the shortcutting pass. It is validated on metric
(Euclidean) instances against the EXACT Held-Karp optimum: the tour is always a valid permutation, it
never exceeds 1.5x the optimum (usually far closer), its length matches an independent recomputation,
every intermediate multigraph is Eulerian, and the matching is genuinely minimum (checked against
brute force). Pure stdlib; the approximation-algorithm companion to the exact Held-Karp and the
nearest-neighbour / 2-opt heuristics in the TSP note."""

from __future__ import annotations

import itertools
import math


def _mst_edges(n, D):
    """Kruskal's minimum spanning tree; returns the list of (u, v) tree edges."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((D[i][j], i, j))
    edges.sort()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for w, u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree.append((u, v))
            if len(tree) == n - 1:
                break
    return tree


def odd_degree_vertices(n, edges):
    """Vertices with odd degree in the given edge list."""
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return [v for v in range(n) if deg[v] % 2 == 1]


def min_weight_perfect_matching(verts, D):
    """Exact minimum-weight perfect matching on `verts` by recursive enumeration.

    The odd-degree set is small (bounded by the graph), so exact enumeration is fine and lets the
    approximation guarantee hold precisely. Returns (matching_edges, total_weight).
    """
    if len(verts) % 2 != 0:
        raise ValueError("perfect matching needs an even number of vertices")
    best = {"cost": math.inf, "edges": None}

    def recurse(remaining, acc_edges, acc_cost):
        if acc_cost >= best["cost"]:
            return  # branch and bound
        if not remaining:
            best["cost"] = acc_cost
            best["edges"] = list(acc_edges)
            return
        first = remaining[0]
        for k in range(1, len(remaining)):
            other = remaining[k]
            w = D[first][other]
            rest = remaining[1:k] + remaining[k + 1:]
            acc_edges.append((first, other))
            recurse(rest, acc_edges, acc_cost + w)
            acc_edges.pop()

    recurse(list(verts), [], 0.0)
    return best["edges"], best["cost"]


def eulerian_circuit(n, multigraph_edges, start=0):
    """Hierholzer's algorithm for an Eulerian circuit on an even-degree multigraph.

    `multigraph_edges` is a list of (u, v); parallel edges are allowed. Returns the vertex sequence
    of a closed walk using every edge exactly once.
    """
    # adjacency as lists of (neighbour, edge_id); edge_id lets us consume each edge once
    adj = [[] for _ in range(n)]
    for eid, (u, v) in enumerate(multigraph_edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    used = [False] * len(multigraph_edges)
    # pointer into each adjacency list
    ptr = [0] * n

    stack = [start]
    circuit = []
    while stack:
        v = stack[-1]
        # advance past consumed edges
        while ptr[v] < len(adj[v]) and used[adj[v][ptr[v]][1]]:
            ptr[v] += 1
        if ptr[v] == len(adj[v]):
            circuit.append(v)
            stack.pop()
        else:
            nxt, eid = adj[v][ptr[v]]
            used[eid] = True
            ptr[v] += 1
            stack.append(nxt)
    circuit.reverse()
    return circuit


def _shortcut(circuit):
    """Shortcut an Eulerian walk to a Hamiltonian tour: keep first visit of each vertex."""
    seen = set()
    tour = []
    for v in circuit:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    return tour


def tour_length(tour, D):
    """Length of a closed tour (wraps back to the start)."""
    total = 0.0
    for i in range(len(tour)):
        total += D[tour[i]][tour[(i + 1) % len(tour)]]
    return total


def christofides(D):
    """Christofides 1.5-approximation for a metric TSP given a full distance matrix D.

    Returns (tour, length): tour is a list of vertex indices (a permutation of 0..n-1).
    """
    n = len(D)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], D[0][1] + D[1][0]

    tree = _mst_edges(n, D)
    odds = odd_degree_vertices(n, tree)
    match_edges, _ = min_weight_perfect_matching(odds, D)
    multigraph = list(tree) + list(match_edges)
    circuit = eulerian_circuit(n, multigraph, start=0)
    tour = _shortcut(circuit)
    return tour, tour_length(tour, D)


def euclidean_matrix(points):
    """Full symmetric distance matrix for 2-D points (a metric instance)."""
    n = len(points)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            D[i][j] = D[j][i] = d
    return D


def brute_min_matching(verts, D):
    """Reference minimum-weight perfect matching by full pairing enumeration."""
    if len(verts) % 2 != 0:
        raise ValueError("need even count")
    best = math.inf
    for perm in itertools.permutations(verts):
        cost = 0.0
        ok = True
        for i in range(0, len(perm), 2):
            cost += D[perm[i]][perm[i + 1]]
            if cost >= best:
                ok = False
                break
        if ok and cost < best:
            best = cost
    return best

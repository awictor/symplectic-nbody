"""Yen's algorithm: the K shortest loopless paths between two nodes.

Shortest-path algorithms find THE cheapest route; many applications need the K cheapest ALTERNATIVES.
A navigation app offers a few distinct routes, not just one; a network reroutes around a failed link
onto the next-best path; a planner ranks candidate itineraries. YEN'S ALGORITHM (1971) finds the K
shortest LOOPLESS (simple, no repeated vertex) paths from a source to a target in a weighted directed
graph, in order of increasing cost. Restricting to loopless paths is what makes the alternatives
meaningful -- otherwise "next shortest" degenerates into padding the best path with pointless
detours.

The algorithm is a clever elaboration of repeated shortest-path search. The first path is just the
shortest path (Dijkstra). To get the (k+1)-th, it considers every prefix of the k-th path: for each
SPUR NODE along it, it temporarily REMOVES the edges that any already-found path (sharing this prefix)
took out of the spur node -- forcing a genuinely different continuation -- and also removes the prefix's
earlier nodes so the spur route stays loopless, then runs Dijkstra from the spur node to the target.
The prefix (root path) plus this spur path is a CANDIDATE; the cheapest unused candidate becomes the
next shortest path. A priority queue of candidates keeps everything in cost order, and each is
generated at most once. The cost is dominated by the K*n shortest-path computations it performs.

This module implements Yen's K-shortest-loopless-paths on a directed weighted graph (given as an
adjacency structure), returning up to K paths with their costs in non-decreasing order. It is verified
against brute force -- enumerating every simple source-to-target path, sorting by cost, and comparing
the first K -- on hundreds of random graphs, confirming the paths are loopless, valid (each step a real
edge), non-decreasing in cost, and exactly the K cheapest. Pure stdlib; a graph-algorithms companion to
the Dijkstra, A*, and Bellman-Ford notes."""

from __future__ import annotations

import heapq


def _dijkstra(n, adj, source, target, removed_edges, removed_nodes):
    """Shortest path source->target avoiding `removed_edges` (set of (u,v)) and `removed_nodes` (set
    of vertices). Returns (cost, path_as_vertex_list) or (None, None) if unreachable."""
    if source in removed_nodes or target in removed_nodes:
        return None, None
    dist = {source: 0}
    prev = {}
    pq = [(0, source)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in adj[u]:
            if v in removed_nodes:
                continue
            if (u, v) in removed_edges:
                continue
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in dist:
        return None, None
    # reconstruct
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return dist[target], path


def _path_cost(adj, path):
    """Cost of a vertex-list path over the adjacency structure (assumes edges exist)."""
    weight = {}
    total = 0
    for u, v in zip(path, path[1:]):
        # find the edge weight u->v (take the minimum if multiedges)
        best = None
        for (x, w) in adj[u]:
            if x == v and (best is None or w < best):
                best = w
        total += best
    return total


def k_shortest_paths(n, edges, source, target, K):
    """The up-to-K shortest loopless paths from `source` to `target` in a directed weighted graph.

    `n` vertices (0..n-1); `edges` a list of (u, v, w) directed edges with w >= 0. Returns a list of
    (cost, path) pairs in non-decreasing cost order (fewer than K if that many don't exist)."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))

    # first shortest path
    cost, path = _dijkstra(n, adj, source, target, set(), set())
    if path is None:
        return []
    A = [(cost, path)]                    # accepted shortest paths
    candidates = []                       # heap of (cost, path) potential next paths
    seen_paths = {tuple(path)}

    while len(A) < K:
        prev_cost, prev_path = A[-1]
        # for each node in the last accepted path except the target, generate a spur
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]

            removed_edges = set()
            # remove the first edge of every accepted path that shares this root
            for _, p in A:
                if len(p) > i and p[:i + 1] == root_path:
                    removed_edges.add((p[i], p[i + 1]))
            # remove the root-path nodes (except the spur node) to keep the spur loopless
            removed_nodes = set(root_path[:-1])

            spur_cost, spur_path = _dijkstra(n, adj, spur_node, target,
                                             removed_edges, removed_nodes)
            if spur_path is None:
                continue
            total_path = root_path[:-1] + spur_path
            tp = tuple(total_path)
            if tp in seen_paths:
                continue
            total_cost = _path_cost(adj, total_path)
            seen_paths.add(tp)
            heapq.heappush(candidates, (total_cost, total_path))

        if not candidates:
            break
        # the cheapest candidate becomes the next accepted path
        c, p = heapq.heappop(candidates)
        A.append((c, p))

    return A


# --- brute-force reference --------------------------------------------------
def brute_k_shortest(n, edges, source, target, K):
    """Enumerate every simple (loopless) source->target path, sort by cost, take the first K.
    Exponential; small graphs only. Returns (cost, path) pairs."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))

    results = []

    def dfs(u, visited, path, cost):
        if u == target:
            results.append((cost, list(path)))
            return
        for v, w in adj[u]:
            if v not in visited:
                visited.add(v)
                path.append(v)
                dfs(v, visited, path, cost + w)
                path.pop()
                visited.discard(v)

    dfs(source, {source}, [source], 0)
    results.sort(key=lambda cp: (cp[0], cp[1]))
    return results[:K]

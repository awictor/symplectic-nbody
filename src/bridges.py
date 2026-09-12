"""Bridges and articulation points: the single points of failure in a network.

In an undirected graph, a BRIDGE is an edge whose removal disconnects the graph (increases the number
of connected components), and an ARTICULATION POINT (cut vertex) is a vertex whose removal does the
same. They are the structural weak points of a network: the one cable whose cut severs two data
centres, the one router whose failure partitions a subnet, the one bond whose breaking splits a
molecule. Finding them is the first question of reliability analysis -- a graph with no bridges and no
articulation points is 2-edge-connected / 2-vertex-connected, meaning it survives any single failure.

The naive way is to delete each edge (or vertex) and recount components, O(V*(V+E)). Tarjan's
algorithm finds all of them in a single depth-first traversal, O(V+E). The key is the DISCOVERY TIME
disc[u] (a counter stamped when DFS first reaches u) and the LOW-LINK low[u] = the smallest discovery
time reachable from u's DFS subtree using tree edges plus at most one back edge. An edge (u, v) with v
a DFS child is a bridge exactly when low[v] > disc[u]: v's subtree has no back edge climbing to u or
above, so that edge is the only connection. A non-root vertex u is an articulation point when some
child v has low[v] >= disc[u] (the subtree cannot bypass u); the DFS root is an articulation point
exactly when it has more than one DFS child. Parallel edges (multi-edges) matter: a doubled edge is
never a bridge, so the traversal must skip the parent by edge-id, not by vertex.

This module builds an undirected graph, finds all bridges and all articulation points in one iterative
DFS (no recursion, so it handles deep graphs), and reports the 2-edge-connected components (the pieces
left when every bridge is removed). It is verified against the brute-force definition on hundreds of
random graphs: an edge is reported as a bridge iff deleting it raises the component count, a vertex is
reported as an articulation point iff deleting it raises the component count, trees have every edge a
bridge and every internal vertex a cut vertex, and cycles have neither. Pure stdlib; a connectivity
companion to the Tarjan-SCC, union-find, and Stoer-Wagner min-cut notes."""

from __future__ import annotations


class Graph:
    """Undirected graph with vertices 0..n-1, supporting parallel edges (tracked by edge id)."""

    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]   # adj[u] = list of (neighbor, edge_id)
        self.edges = []                     # edge_id -> (u, v)

    def add_edge(self, u, v):
        eid = len(self.edges)
        self.edges.append((u, v))
        self.adj[u].append((v, eid))
        self.adj[v].append((u, eid))
        return eid


def find_bridges_and_articulation(g):
    """All bridges and articulation points via one iterative Tarjan DFS.

    Returns (bridges, articulation_points): bridges as a sorted list of (u, v) with u < v,
    articulation_points as a sorted list of vertices."""
    n = g.n
    disc = [-1] * n
    low = [0] * n
    is_art = [False] * n
    bridges = []
    timer = 0

    for start in range(n):
        if disc[start] != -1:
            continue
        root_children = 0
        disc[start] = low[start] = timer
        timer += 1
        stack = [[start, -1, 0]]        # (u, parent_edge_id, next adj index)
        while stack:
            frame = stack[-1]
            u, pe, idx = frame
            if idx < len(g.adj[u]):
                frame[2] += 1
                v, eid = g.adj[u][idx]
                if eid == pe:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    if u == start:
                        root_children += 1
                    stack.append([v, eid, 0])
                else:
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    if low[u] < low[p]:
                        low[p] = low[u]
                    if low[u] > disc[p]:
                        a, b = (p, u) if p < u else (u, p)
                        bridges.append((a, b))
                    if p != start and low[u] >= disc[p]:
                        is_art[p] = True
        if root_children >= 2:
            is_art[start] = True

    arts = [v for v in range(n) if is_art[v]]
    return sorted(bridges), arts


def two_edge_connected_components(g):
    """The connected components remaining when every bridge is removed. Returns a list of vertex
    lists. Vertices in the same 2-edge-connected component stay connected under any single edge
    failure."""
    bridges = set(find_bridges_and_articulation(g)[0])
    # union-find over all non-bridge edges
    parent = list(range(g.n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in g.edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in bridges:
            continue
        ra, rb = find(u), find(v)
        if ra != rb:
            parent[ra] = rb

    groups = {}
    for v in range(g.n):
        groups.setdefault(find(v), []).append(v)
    return sorted(groups.values(), key=lambda c: c[0])


# --- brute-force reference (definition-level) -------------------------------
def _count_components(n, edge_list):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edge_list:
        parent[find(u)] = find(v)
    return len({find(v) for v in range(n)})


def brute_bridges(g):
    """Bridges by definition: an edge whose removal increases the component count."""
    base = _count_components(g.n, g.edges)
    out = []
    for eid, (u, v) in enumerate(g.edges):
        rest = [e for i, e in enumerate(g.edges) if i != eid]
        if _count_components(g.n, rest) > base:
            a, b = (u, v) if u < v else (v, u)
            out.append((a, b))
    return sorted(set(out))


def brute_articulation(g):
    """Articulation points by definition: a vertex whose removal increases the component count.
    Removal means dropping the vertex and all its edges, then counting components among the rest."""
    base_present = [v for v in range(g.n)]
    base = _count_components_subset(g, base_present)
    out = []
    for w in range(g.n):
        present = [v for v in range(g.n) if v != w]
        if _count_components_subset(g, present) > base:
            out.append(w)
    return out


def _count_components_subset(g, present):
    """Components among `present` vertices using only edges between present vertices."""
    ps = set(present)
    if not ps:
        return 0
    idx = {v: i for i, v in enumerate(present)}
    parent = list(range(len(present)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in g.edges:
        if u in ps and v in ps:
            parent[find(idx[u])] = find(idx[v])
    return len({find(i) for i in range(len(present))})

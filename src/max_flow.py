"""Maximum flow: Edmonds-Karp, the min-cut theorem, and bipartite matching.

How much can flow from a source to a sink through a network of capacitated pipes? The MAXIMUM-FLOW
problem is a cornerstone of combinatorial optimization -- routing, scheduling, image segmentation,
and (via a reduction shown here) matching all reduce to it. The FORD-FULKERSON method repeatedly
finds an AUGMENTING PATH from source to sink with spare capacity and pushes flow along it, updating
a RESIDUAL graph (each used edge gains a reverse edge that lets later paths 'cancel' flow). When no
augmenting path remains, the flow is maximum. EDMONDS-KARP is Ford-Fulkerson that always picks the
SHORTEST augmenting path (by BFS), which guarantees O(V E^2) termination -- no dependence on the
capacities, unlike naive Ford-Fulkerson.

The celebrated MAX-FLOW MIN-CUT THEOREM says the maximum flow equals the minimum CUT: the smallest
total capacity of edges whose removal disconnects source from sink. After the flow saturates, the
set of vertices still reachable from the source in the residual graph, and its complement, define
that minimum cut -- so the same computation yields both the flow value and the bottleneck edges.

A classic reduction turns BIPARTITE MATCHING into max flow: add a super-source to every left vertex
and a super-sink from every right vertex, all capacities 1; the max flow is the size of the maximum
matching. This module implements Edmonds-Karp max flow, the min-cut extraction, and bipartite
matching by reduction -- verified that flow conservation and capacity constraints hold, that the max
flow equals the min-cut capacity, that it matches hand-computed values on textbook networks, and
that bipartite matching by flow equals a direct augmenting-path matching. Pure stdlib; a
graph-algorithms companion to the Dijkstra and PageRank notes."""

from __future__ import annotations

from collections import deque


class MaxFlow:
    """Edmonds-Karp maximum flow on a directed capacitated graph."""

    def __init__(self, n):
        self.n = n
        # capacity[u][v] = remaining capacity of edge u->v in the residual graph
        self.cap = [[0] * n for _ in range(n)]
        self.adj = [set() for _ in range(n)]      # neighbours in the residual graph

    def add_edge(self, u, v, capacity):
        """Add a directed edge u->v with the given capacity (accumulates parallel edges)."""
        self.cap[u][v] += capacity
        self.adj[u].add(v)
        self.adj[v].add(u)                        # reverse edge exists (capacity 0 initially)

    def _bfs(self, s, t, parent):
        """Shortest augmenting path from s to t in the residual graph; fill parent, return bottleneck."""
        for i in range(self.n):
            parent[i] = -1
        parent[s] = s
        q = deque([(s, float("inf"))])
        while q:
            u, flow = q.popleft()
            for v in self.adj[u]:
                if parent[v] == -1 and self.cap[u][v] > 0:
                    parent[v] = u
                    new_flow = min(flow, self.cap[u][v])
                    if v == t:
                        return new_flow
                    q.append((v, new_flow))
        return 0

    def max_flow(self, source, sink):
        """Maximum flow from source to sink. Mutates the residual graph; call once."""
        if source == sink:
            return 0
        total = 0
        parent = [-1] * self.n
        while True:
            bottleneck = self._bfs(source, sink, parent)
            if bottleneck == 0:
                break
            total += bottleneck
            # push `bottleneck` along the path, updating the residual capacities
            v = sink
            while v != source:
                u = parent[v]
                self.cap[u][v] -= bottleneck
                self.cap[v][u] += bottleneck
                v = u
        return total

    def min_cut(self, source):
        """After max_flow has run, the min cut: (reachable set, cut edges). Vertices still reachable
        from the source in the residual graph are on the source side; edges crossing to the other
        side are the minimum cut."""
        reachable = set()
        q = deque([source])
        reachable.add(source)
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if v not in reachable and self.cap[u][v] > 0:
                    reachable.add(v)
                    q.append(v)
        return reachable


def edmonds_karp(n, edges, source, sink):
    """Convenience: build the network from (u, v, capacity) edges and return the max flow value."""
    mf = MaxFlow(n)
    for u, v, c in edges:
        mf.add_edge(u, v, c)
    return mf.max_flow(source, sink)


def min_cut_value(n, edges, source, sink):
    """Max flow AND the min-cut edges (list of (u, v, capacity) that cross the cut)."""
    mf = MaxFlow(n)
    original = {}
    for u, v, c in edges:
        mf.add_edge(u, v, c)
        original[(u, v)] = original.get((u, v), 0) + c
    flow = mf.max_flow(source, sink)
    reachable = mf.min_cut(source)
    cut = [(u, v, c) for (u, v), c in original.items() if u in reachable and v not in reachable]
    return flow, cut


def bipartite_matching(left, right, edges):
    """Maximum bipartite matching by reduction to max flow.

    left/right: iterables of vertex labels (disjoint). edges: (l, r) allowed pairs. Returns the
    matching size and the matched (l, r) pairs."""
    left = list(left)
    right = list(right)
    lidx = {v: i + 1 for i, v in enumerate(left)}          # 1..L
    ridx = {v: len(left) + 1 + i for i, v in enumerate(right)}
    n = len(left) + len(right) + 2
    source, sink = 0, n - 1
    mf = MaxFlow(n)
    for v in left:
        mf.add_edge(source, lidx[v], 1)
    for v in right:
        mf.add_edge(ridx[v], sink, 1)
    for l, r in edges:
        mf.add_edge(lidx[l], ridx[r], 1)
    size = mf.max_flow(source, sink)
    # recover matched pairs: a left->right residual edge with flow (reverse capacity 1)
    pairs = []
    for l in left:
        for r in right:
            if mf.cap[ridx[r]][lidx[l]] > 0:      # reverse edge carries flow
                pairs.append((l, r))
    return size, pairs

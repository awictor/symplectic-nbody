"""Dinic's algorithm: maximum flow by blocking flows on a layered graph.

The MAXIMUM FLOW problem asks how much can be pushed from a source to a sink through a network of
capacitated edges -- the throughput of a pipe system, the bandwidth of a network, the size of a maximum
bipartite matching. The Ford-Fulkerson family augments one path at a time; Edmonds-Karp picks the
shortest augmenting path by BFS and runs in O(V*E^2). DINIC'S ALGORITHM (1970) is markedly faster,
O(V^2*E) in general and O(E*sqrt(V)) on unit-capacity graphs (making it the standard engine for
bipartite matching), by augmenting MANY paths at once per phase.

Each phase has two steps. First a BFS from the source builds the LEVEL GRAPH: level[v] is the shortest
number of edges from the source in the residual network, and only edges going from level L to level L+1
are kept. Then a DFS finds a BLOCKING FLOW -- it repeatedly pushes flow along level-respecting paths
until no source-to-sink path remains in the level graph, saturating at least one edge on each. A key
optimisation is the ITERATION POINTER (`it[v]`): once an edge out of v leads to a dead end in this
phase, it is skipped on all future DFS attempts, so the blocking flow is found in O(V*E). Because the
sink's level strictly increases every phase, only O(V) phases run, giving the overall bound. Each edge
carries a residual capacity and a reverse edge (capacity grows as flow is pushed) exactly as in every
augmenting-path method.

This module builds a flow network with integer capacities, computes the maximum source-to-sink flow via
Dinic's blocking-flow phases, and recovers the minimum cut (the saturated edges separating the source's
residual-reachable set from the rest). It is verified against an independent Edmonds-Karp reference and
the max-flow/min-cut theorem -- the flow value equals the min-cut capacity -- on hundreds of random
networks, plus a bipartite-matching reduction whose max flow equals the matching size. Pure stdlib; a
graph-optimisation companion to the max-flow (Edmonds-Karp), min-cost-flow, and Stoer-Wagner notes."""

from __future__ import annotations

from collections import deque


class Dinic:
    """Max-flow network with integer capacities. Vertices are 0..n-1."""

    def __init__(self, n):
        self.n = n
        # each edge: [to, capacity, index_of_reverse_edge]
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, capacity):
        """Add a directed edge u->v with the given capacity (a reverse residual edge of capacity 0 is
        created automatically). For an undirected edge, add it in both directions."""
        self.graph[u].append([v, capacity, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    def _bfs(self, source, sink):
        """Build the level graph; return True if the sink is reachable in the residual network."""
        self.level = [-1] * self.n
        self.level[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for to, cap, _ in self.graph[u]:
                if cap > 0 and self.level[to] == -1:
                    self.level[to] = self.level[u] + 1
                    q.append(to)
        return self.level[sink] != -1

    def _dfs(self, u, sink, pushed):
        """Push up to `pushed` units along level-respecting paths from u; return the amount pushed."""
        if u == sink:
            return pushed
        while self.it[u] < len(self.graph[u]):
            edge = self.graph[u][self.it[u]]
            to, cap, rev = edge
            if cap > 0 and self.level[to] == self.level[u] + 1:
                d = self._dfs(to, sink, min(pushed, cap))
                if d > 0:
                    edge[1] -= d
                    self.graph[to][rev][1] += d
                    return d
            self.it[u] += 1        # this edge is exhausted for the rest of the phase
        return 0

    def max_flow(self, source, sink):
        """The maximum flow from source to sink. O(V^2 * E)."""
        if source == sink:
            return 0
        flow = 0
        while self._bfs(source, sink):
            self.it = [0] * self.n
            while True:
                f = self._dfs(source, sink, float("inf"))
                if f == 0:
                    break
                flow += f
        return flow

    def min_cut(self, source):
        """After computing the max flow, the set of vertices reachable from the source in the residual
        graph -- one side of the minimum cut."""
        reachable = [False] * self.n
        q = deque([source])
        reachable[source] = True
        while q:
            u = q.popleft()
            for to, cap, _ in self.graph[u]:
                if cap > 0 and not reachable[to]:
                    reachable[to] = True
                    q.append(to)
        return [v for v in range(self.n) if reachable[v]]


def max_flow(n, edges, source, sink):
    """Convenience wrapper: maximum flow of a directed network given as (u, v, capacity) edges."""
    d = Dinic(n)
    for u, v, c in edges:
        d.add_edge(u, v, c)
    return d.max_flow(source, sink)


def min_cut_value(n, edges, source, sink):
    """The minimum-cut capacity (equal to the max flow) and the source side of the cut."""
    d = Dinic(n)
    for u, v, c in edges:
        d.add_edge(u, v, c)
    flow = d.max_flow(source, sink)
    return flow, d.min_cut(source)


def bipartite_matching_size(n_left, n_right, pairs):
    """Maximum bipartite matching size via a max-flow reduction: source -> each left (cap 1),
    each allowed (left,right) pair (cap 1), each right -> sink (cap 1)."""
    # node ids: 0 = source, 1..n_left = left, n_left+1..n_left+n_right = right, last = sink
    S = 0
    T = n_left + n_right + 1
    d = Dinic(n_left + n_right + 2)
    for i in range(n_left):
        d.add_edge(S, 1 + i, 1)
    for j in range(n_right):
        d.add_edge(1 + n_left + j, T, 1)
    for (li, rj) in pairs:
        d.add_edge(1 + li, 1 + n_left + rj, 1)
    return d.max_flow(S, T)


# --- brute-force / independent reference ------------------------------------
def edmonds_karp(n, edges, source, sink):
    """Independent max-flow reference: BFS shortest augmenting paths (Edmonds-Karp)."""
    cap = [[0] * n for _ in range(n)]
    adj = [[] for _ in range(n)]
    for u, v, c in edges:
        if cap[u][v] == 0 and cap[v][u] == 0:
            adj[u].append(v)
            adj[v].append(u)
        cap[u][v] += c
    flow = 0
    while True:
        parent = [-1] * n
        parent[source] = source
        q = deque([source])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    q.append(v)
        if parent[sink] == -1:
            break
        # bottleneck
        aug = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            aug = min(aug, cap[u][v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            cap[u][v] -= aug
            cap[v][u] += aug
            v = u
        flow += aug
    return flow

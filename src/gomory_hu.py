"""Gomory-Hu tree: all-pairs minimum cuts of an undirected graph from just n-1 max-flow calls.

An undirected weighted graph on n vertices has C(n, 2) pairs, and each pair (s, t) has a minimum
cut -- the least total edge weight you must remove to separate them. Computing all of them the
obvious way is C(n, 2) max-flow computations. The remarkable theorem of Gomory and Hu (1961) is
that only n-1 of those cuts are ever distinct, and they can be organised into a single weighted
TREE on the same vertex set with a beautiful property:

    for ANY pair (s, t), the minimum cut between them in the original graph equals the MINIMUM
    edge weight on the unique tree path from s to t.

So the whole all-pairs min-cut table -- quadratically many numbers -- is encoded in a tree with
n-1 edges, and any query is a path-minimum. The tree also tells you the cut itself: removing the
lightest edge on the s-t path splits the tree into two components, and that vertex bipartition is
an actual minimum s-t cut of the graph.

This module builds the tree with Gusfield's simplified algorithm, which avoids the vertex
contractions of the original construction and instead runs n-1 ordinary s-t max-flow computations
on the UNCHANGED graph, wiring up a parent pointer per vertex:

    parent[i] starts at 0 for every i.
    For i = 1 .. n-1:
        f, S = min s-t cut between i and parent[i]     (S = the side containing i)
        weight[i] = f
        for every j > i with j in S and parent[j] == parent[i]:  parent[j] = i
        if parent[parent[i]] in S:                     (the reparent fix-up)
            parent[i], parent[parent[i]] = parent[parent[i]], i
            weight[i], weight[parent[i]] = weight[parent[i]], f

The resulting edges (i, parent[i], weight[i]) form the Gomory-Hu tree. Each max-flow here uses
Dinic on the undirected graph (each edge added in both directions).

Validated exhaustively: on random graphs the tree's path-minimum equals a brute-force independent
s-t min cut (computed by a separate max-flow) for EVERY pair, and the bipartition induced by the
lightest path edge really has that cut weight. Pure stdlib; the all-pairs companion to the single
global min cut of Stoer-Wagner and the single s-t cut of Dinic."""

from __future__ import annotations


# --- self-contained Dinic max flow (undirected s-t min cut) ------------------
class _Dinic:
    def __init__(self, n):
        self.n = n
        self.to = []
        self.cap = []
        self.head = [[] for _ in range(n)]

    def add_undirected(self, u, v, c):
        # an undirected edge of capacity c: both residual arcs start at c
        self.head[u].append(len(self.to)); self.to.append(v); self.cap.append(c)
        self.head[v].append(len(self.to)); self.to.append(u); self.cap.append(c)

    def _bfs(self, s, t):
        self.level = [-1] * self.n
        self.level[s] = 0
        q = [s]
        while q:
            nxt = []
            for u in q:
                for e in self.head[u]:
                    v = self.to[e]
                    if self.cap[e] > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        nxt.append(v)
            q = nxt
        return self.level[t] >= 0

    def _dfs(self, u, t, pushed):
        if u == t:
            return pushed
        while self.it[u] < len(self.head[u]):
            e = self.head[u][self.it[u]]
            v = self.to[e]
            if self.cap[e] > 0 and self.level[v] == self.level[u] + 1:
                d = self._dfs(v, t, min(pushed, self.cap[e]))
                if d > 0:
                    self.cap[e] -= d
                    self.cap[e ^ 1] += d
                    return d
            self.it[u] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0
        while self._bfs(s, t):
            self.it = [0] * self.n
            while True:
                f = self._dfs(s, t, float("inf"))
                if f == 0:
                    break
                flow += f
        return flow

    def reachable(self, s):
        """Vertices reachable from s in the residual graph = the source side of the min cut."""
        seen = [False] * self.n
        seen[s] = True
        q = [s]
        while q:
            u = q.pop()
            for e in self.head[u]:
                v = self.to[e]
                if self.cap[e] > 0 and not seen[v]:
                    seen[v] = True
                    q.append(v)
        return seen


def min_cut_st(n, edges, s, t):
    """Minimum s-t cut of an undirected graph. Returns (weight, source_side_set)."""
    d = _Dinic(n)
    for u, v, w in edges:
        d.add_undirected(u, v, w)
    f = d.max_flow(s, t)
    seen = d.reachable(s)
    side = frozenset(i for i in range(n) if seen[i])
    return f, side


# --- Gomory-Hu tree via Gusfield's algorithm ---------------------------------
def gomory_hu_tree(n, edges):
    """Build the Gomory-Hu tree. Returns a list of (u, v, weight) tree edges (n-1 of them,
    or fewer if the graph is disconnected -- a disconnected pair gets a weight-0 tree edge)."""
    if n == 0:
        return []
    parent = [0] * n
    weight = [0] * n
    for i in range(1, n):
        f, side = min_cut_st(n, edges, i, parent[i])
        weight[i] = f
        pi = parent[i]
        for j in range(i + 1, n):
            if j in side and parent[j] == pi:
                parent[j] = i
        # Gusfield reparent fix-up: if parent[pi] is on i's side, rotate the edge
        if parent[pi] in side:
            parent[i] = parent[pi]
            parent[pi] = i
            weight[i] = weight[pi]
            weight[pi] = f
    return [(i, parent[i], weight[i]) for i in range(1, n)]


def _tree_adj(n, tree_edges):
    adj = [[] for _ in range(n)]
    for u, v, w in tree_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def min_cut_query(n, tree_edges, s, t):
    """Minimum s-t cut = the lightest edge on the unique tree path from s to t."""
    if s == t:
        return 0
    adj = _tree_adj(n, tree_edges)
    # BFS carrying the running minimum edge weight along the path
    best = [None] * n
    best[s] = float("inf")
    prev = [-1] * n
    q = [s]
    while q:
        u = q.pop()
        if u == t:
            break
        for v, w in adj[u]:
            if best[v] is None:
                best[v] = min(best[u], w)
                prev[v] = u
                q.append(v)
    return best[t] if best[t] is not None and best[t] != float("inf") else 0


def all_pairs_min_cuts(n, tree_edges):
    """The full symmetric all-pairs min-cut table, read off the tree in O(n^2)."""
    table = [[0] * n for _ in range(n)]
    for s in range(n):
        for t in range(s + 1, n):
            c = min_cut_query(n, tree_edges, s, t)
            table[s][t] = table[t][s] = c
    return table

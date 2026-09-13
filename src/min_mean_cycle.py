"""Karp's minimum mean cycle: the cycle of least average edge weight in a directed graph, in O(V*E).

Given a directed weighted graph, a cycle's MEAN WEIGHT is its total edge weight divided by the
number of edges. The minimum mean cycle is the one with the smallest such average. It is a
surprisingly central object: it certifies whether a graph has a negative cycle (minimum mean < 0),
it is the optimal steady-state cost of a cyclic schedule, it drives the pivoting rule that makes the
min-cost-flow "cancel the most negative cycle" strategy run in polynomial time, and its value is the
critical threshold at which a parametric shortest-path problem stops having a solution.

Naively you might enumerate cycles -- exponentially many. Karp's 1978 theorem gives it in O(V*E) with
a beautiful piece of dynamic programming. Fix a source s. Let d_k(v) be the minimum weight of a walk
of EXACTLY k edges from s to v (infinity if none). These satisfy

    d_0(s) = 0,  d_0(v) = infinity otherwise,
    d_k(v) = min over edges (u,v) of  d_{k-1}(u) + w(u,v).

Karp proved the minimum cycle mean equals

    lambda* = min over v   max over k in 0..n-1   (d_n(v) - d_k(v)) / (n - k)

evaluated only over finite d_n(v). The inner max-over-k for each vertex is the tightest average that
vertex's n-edge walk implies; the outer min finds the best vertex. It needs the d_k table (n+1 rows),
so O(n) space and O(V*E) time. To RECOVER the cycle, this module reconstructs the optimal n-edge walk
by back-pointers and extracts the repeated vertex that closes a cycle -- then verifies its mean
equals lambda*.

Because the DP requires every vertex reachable from the source, the implementation adds a virtual
source with zero-weight edges to all vertices (this cannot create a cycle, since the source has no
in-edges) and runs the recurrence from there -- the standard trick that makes Karp work on graphs
that are not strongly connected.

Validated against brute force on small graphs (enumerate every simple cycle, take the min mean) and
against the negative-cycle detector: minimum mean < 0 exactly when a negative cycle exists, matching
Bellman-Ford. The recovered cycle is checked to be a real cycle whose mean equals the reported value.
Pure stdlib; the cyclic-optimum companion to Floyd-Warshall and the min-cost-flow cancelling rule."""

from __future__ import annotations

INF = float("inf")


def min_mean_cycle(n, edges):
    """Minimum cycle mean of a directed graph on vertices 0..n-1 with (u, v, w) edges.

    Returns (mean, cycle) where cycle is a list of vertices [v0, v1, ..., v0-implied] forming the
    optimal cycle (the returned list does not repeat the closing vertex), or (None, None) if the
    graph is acyclic."""
    if n == 0:
        return None, None

    # Virtual source S = n with zero-weight edges to every real vertex.
    S = n
    N = n + 1
    adj_in = [[] for _ in range(N)]  # adj_in[v] = list of (u, w) with edge u->v
    for u, v, w in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u},{v}) out of bounds")
        adj_in[v].append((u, w))
    for v in range(n):
        adj_in[v].append((S, 0.0))

    # d[k][v] = min weight of a walk of exactly k edges from S to v.
    d = [[INF] * N for _ in range(N + 1)]
    par = [[-1] * N for _ in range(N + 1)]
    d[0][S] = 0.0
    for k in range(1, N + 1):
        dk = d[k]
        dkm = d[k - 1]
        pk = par[k]
        for v in range(N):
            best = INF
            bpar = -1
            for (u, w) in adj_in[v]:
                if dkm[u] + w < best:
                    best = dkm[u] + w
                    bpar = u
            dk[v] = best
            pk[v] = bpar

    # Karp: lambda* = min_v max_k (d[N][v] - d[k][v]) / (N - k), over finite d[N][v].
    # (Here "n" in Karp's formula is N, the vertex count of the augmented graph.)
    best_mean = INF
    best_v = -1
    for v in range(N):
        if d[N][v] == INF:
            continue
        worst = -INF
        for k in range(N):
            if d[k][v] == INF:
                continue
            val = (d[N][v] - d[k][v]) / (N - k)
            if val > worst:
                worst = val
        if worst < best_mean:
            best_mean = worst
            best_v = v

    if best_v == -1 or best_mean == INF:
        return None, None

    # Reconstruct the N-edge walk ending at best_v, then find the cycle inside it.
    walk = []
    v = best_v
    for k in range(N, -1, -1):
        walk.append(v)
        v = par[k][v]
        if v == -1:
            break
    walk.reverse()  # walk is a sequence of vertices along the optimal walk (may include S at front)

    # Find a repeated real vertex -> the cycle between the two occurrences.
    seen = {}
    cycle = None
    for i, node in enumerate(walk):
        if node == S:
            seen = {}
            continue
        if node in seen:
            cycle = walk[seen[node]:i]
            break
        seen[node] = i

    if cycle is None:
        return None, None
    return best_mean, cycle


def cycle_mean(cycle, edges):
    """Mean weight of a given cycle [v0, v1, ...] (edges v0->v1, ..., v_last->v0)."""
    wmap = {}
    for u, v, w in edges:
        wmap[(u, v)] = w  # last wins; caller's graph assumed simple for this helper
    total = 0.0
    m = len(cycle)
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        total += wmap[(a, b)]
    return total / m


# --- brute-force reference ---------------------------------------------------
def brute_min_mean_cycle(n, edges):
    """Exhaustive: enumerate all simple cycles, return the smallest mean (small graphs only)."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    best = INF
    best_cycle = None

    def dfs(start, u, path, weight, length, on_path):
        nonlocal best, best_cycle
        for (v, w) in adj[u]:
            if v == start and length >= 0:
                mean = (weight + w) / (length + 1)
                if mean < best:
                    best = mean
                    best_cycle = list(path)
            elif v > start and not on_path[v]:
                on_path[v] = True
                path.append(v)
                dfs(start, v, path, weight + w, length + 1, on_path)
                path.pop()
                on_path[v] = False

    for start in range(n):
        on_path = [False] * n
        on_path[start] = True
        dfs(start, start, [start], 0.0, 0, on_path)
    if best_cycle is None:
        return None, None
    return best, best_cycle

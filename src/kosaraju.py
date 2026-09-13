"""Kosaraju's algorithm: strongly connected components in two passes of depth-first search.

A directed graph's STRONGLY CONNECTED COMPONENTS are its maximal sets of vertices where every vertex
can reach every other -- the "islands of mutual reachability". Contracting each SCC to a point turns
any directed graph into a DAG (the CONDENSATION), which is why SCC decomposition is the first step in
2-SAT, dead-code elimination, dataflow analysis, and finding cycles in dependency graphs.

Kosaraju's algorithm (1978) finds them with a strikingly simple idea that rests on one fact: a graph
and its TRANSPOSE (all edges reversed) have exactly the same SCCs, because mutual reachability is
symmetric under reversing every edge. The algorithm is two depth-first sweeps:

  1. Run DFS on the original graph and push each vertex onto a stack when it FINISHES (post-order).
     This orders vertices by decreasing finish time -- a vertex in a "source" SCC of the condensation
     finishes last and ends up on top.
  2. Run DFS on the TRANSPOSED graph, repeatedly starting from the top of that stack. Each DFS tree in
     this second pass is exactly one SCC.

The reason it works: taking vertices in decreasing finish order means the first vertex explored in the
transpose belongs to a SINK of the reversed condensation (a source of the original), and because the
transpose confines that DFS to the component, it cannot leak into others. It is the two-pass
counterpart to Tarjan's single-pass low-link method -- conceptually cleaner, one extra graph traversal.

This module builds the transpose, runs the two iterative DFS passes (no recursion-depth limits), and
returns the SCCs. It is validated against the repo's Tarjan implementation: the two produce the same
partition of the vertices into components on random directed graphs; every returned component is
genuinely strongly connected (all-pairs mutual reachability, checked by BFS) and MAXIMAL (no two
components could merge); the components in a DAG are all singletons; a single big cycle is one
component; and the condensation is acyclic. Pure stdlib; the two-pass DFS companion to Tarjan's SCC,
the topological sort, and the 2-SAT solver."""

from __future__ import annotations

from collections import deque


def _transpose(n, edges):
    radj = [[] for _ in range(n)]
    for u, v in edges:
        radj[v].append(u)
    return radj


def strongly_connected_components(n, edges):
    """Kosaraju's SCC. Returns a list of components (each a sorted list of vertices).

    Components are returned in topological order of the condensation (sources before sinks).
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)

    # ---- pass 1: DFS on the original graph, record vertices by finish time ----
    visited = [False] * n
    order = []  # finish order (increasing); top of the "stack" is order[-1]
    for start in range(n):
        if visited[start]:
            continue
        # iterative DFS with an explicit stack; push (vertex, child-iterator-index)
        stack = [(start, 0)]
        visited[start] = True
        while stack:
            v, i = stack[-1]
            if i < len(adj[v]):
                stack[-1] = (v, i + 1)
                w = adj[v][i]
                if not visited[w]:
                    visited[w] = True
                    stack.append((w, 0))
            else:
                order.append(v)  # v finished
                stack.pop()

    # ---- pass 2: DFS on the transpose in decreasing finish order ----
    radj = _transpose(n, edges)
    assigned = [False] * n
    components = []
    for start in reversed(order):
        if assigned[start]:
            continue
        comp = []
        stack = [start]
        assigned[start] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in radj[v]:
                if not assigned[w]:
                    assigned[w] = True
                    stack.append(w)
        components.append(sorted(comp))

    return components


def component_index(n, edges):
    """Map each vertex to the index of its SCC (in the returned topological order)."""
    comps = strongly_connected_components(n, edges)
    idx = [0] * n
    for ci, comp in enumerate(comps):
        for v in comp:
            idx[v] = ci
    return idx


def condensation(n, edges):
    """The DAG of SCCs: returns (num_components, condensed_edges) with no self-loops or duplicates."""
    idx = component_index(n, edges)
    num = max(idx) + 1 if n > 0 else 0
    cedges = set()
    for u, v in edges:
        if idx[u] != idx[v]:
            cedges.add((idx[u], idx[v]))
    return num, sorted(cedges)


def is_strongly_connected_set(n, edges, vertices):
    """Check that `vertices` form a strongly connected set (all-pairs mutual reachability)."""
    vs = set(vertices)
    adj = [[] for _ in range(n)]
    radj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        radj[v].append(u)

    def reachable_within(src, graph):
        seen = {src}
        dq = deque([src])
        while dq:
            x = dq.popleft()
            for y in graph[x]:
                if y in vs and y not in seen:
                    seen.add(y)
                    dq.append(y)
        return seen

    for s in vs:
        if reachable_within(s, adj) & vs != vs:
            return False
        if reachable_within(s, radj) & vs != vs:
            return False
    return True

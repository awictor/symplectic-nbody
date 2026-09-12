"""Tarjan's strongly connected components, condensation, and topological order.

In a directed graph, a STRONGLY CONNECTED COMPONENT (SCC) is a maximal set of vertices in which every
vertex can reach every other -- the graph's cycles, collapsed into equivalence classes. Finding the
SCCs reveals a directed graph's deep structure: shrinking each SCC to a single node yields the
CONDENSATION, which is always a DAG (directed acyclic graph), so any directed graph is a DAG of its
cycles. This underlies dead-code and dependency analysis in compilers, deadlock detection, the
computation of 2-SAT, and the block structure of the web graph.

TARJAN'S ALGORITHM finds all SCCs in a single depth-first search in O(V + E) -- optimal. It assigns
each vertex a DISCOVERY INDEX in DFS order and computes a LOW-LINK value: the smallest index
reachable from the vertex's subtree using at most one back-edge to a vertex still on the DFS stack.
A vertex whose low-link equals its own index is the ROOT of an SCC; when the DFS finishes exploring
it, everything pushed onto an auxiliary stack after it (and itself) forms one component. Because the
low-link propagates the reachability of cycles, the roots partition the vertices into exactly the
strongly connected components, and the components are discovered in REVERSE topological order of the
condensation -- so the condensation's topological sort falls out for free.

This module implements Tarjan's SCC (iteratively, to avoid Python recursion limits on large graphs),
the condensation DAG, a topological sort of any DAG (Kahn's algorithm), and a cycle test. It is
verified against a brute-force mutual-reachability check (two vertices share an SCC iff each reaches
the other), that the condensation is acyclic, that the SCCs are returned in reverse topological
order, that a DAG yields all singleton components and a valid topological order, that a single cycle
is one component, and on hand-checked graphs. Pure stdlib; a graph-algorithm companion to the
Dijkstra, max-flow, and union-find notes."""

from __future__ import annotations


def strongly_connected_components(n, edges):
    """Tarjan's SCC. n vertices (0..n-1), edges a list of (u, v) directed edges. Returns a list of
    components (each a list of vertices), in REVERSE topological order of the condensation."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)

    index = [None] * n            # discovery index
    low = [0] * n                 # low-link
    on_stack = [False] * n
    stack = []
    result = []
    counter = [0]

    # iterative DFS to avoid recursion depth limits
    for start in range(n):
        if index[start] is not None:
            continue
        # work stack holds (vertex, next-neighbour-pointer)
        work = [(start, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter[0]
                counter[0] += 1
                stack.append(v)
                on_stack[v] = True
            recursed = False
            for i in range(pi, len(adj[v])):
                w = adj[v][i]
                if index[w] is None:
                    work[-1] = (v, i + 1)     # resume after this neighbour
                    work.append((w, 0))
                    recursed = True
                    break
                elif on_stack[w]:
                    low[v] = min(low[v], index[w])
            if recursed:
                continue
            # done with v: it is an SCC root iff low[v] == index[v]
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                result.append(comp)
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return result


def condensation(n, edges):
    """The condensation DAG. Returns (component_of, dag_edges, components) where component_of[v] is
    the SCC id of vertex v, and dag_edges is the set of (comp_a, comp_b) super-edges."""
    comps = strongly_connected_components(n, edges)
    comp_of = [0] * n
    for cid, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = cid
    dag = set()
    for u, v in edges:
        if comp_of[u] != comp_of[v]:
            dag.add((comp_of[u], comp_of[v]))
    return comp_of, dag, comps


def topological_sort(n, edges):
    """A topological order of a DAG via Kahn's algorithm. Returns the order, or None if the graph has
    a cycle (is not a DAG)."""
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    from collections import deque
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for w in adj[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return order if len(order) == n else None


def has_cycle(n, edges):
    """True if the directed graph contains a cycle."""
    return topological_sort(n, edges) is None


def is_dag(n, edges):
    """True if the directed graph is acyclic."""
    return topological_sort(n, edges) is not None

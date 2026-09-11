"""Topological sort: ordering tasks so every prerequisite comes first.

A directed acyclic graph (DAG) encodes dependencies: an edge u -> v means "u must come before
v". A topological order is a linear arrangement of the nodes in which every edge points forward
-- exactly the order you can do the tasks, compile the modules, install the packages, or run the
spreadsheet cells so nothing is attempted before what it depends on. It exists if and only if
the graph has no cycle (a cycle is a circular dependency that can never be satisfied).

Two classic algorithms both run in O(V + E):

  * KAHN'S algorithm (1962) is breadth-first on in-degrees: repeatedly take a node with no
    remaining incoming edges, output it, and remove its outgoing edges (lowering neighbours'
    in-degrees). If nodes remain but none has in-degree zero, the leftover forms a cycle.

  * The DFS method finishes each node after its descendants and prepends it to the order; a
    back-edge to a node still on the recursion stack reveals a cycle.

Beyond ordering, the same DAG structure gives the CRITICAL PATH: with a duration on each task,
the longest path through the DAG is the minimum time to finish everything (project scheduling,
PERT), and it is computed by relaxing edges in topological order.

This module implements both sorts, cycle detection, and the longest-path / critical-path
computation, and verifies that any order it returns respects every edge. Pure stdlib; the
DAG-ordering companion to the Dijkstra and Union-Find notes.
"""

from __future__ import annotations


class DAG:
    """A directed graph as an adjacency list; nodes are any hashable."""

    def __init__(self):
        self.adj = {}

    def add_node(self, u):
        self.adj.setdefault(u, [])

    def add_edge(self, u, v):
        """Add a directed edge u -> v (u must precede v)."""
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append(v)

    def nodes(self):
        return list(self.adj.keys())

    def edges(self):
        return [(u, v) for u in self.adj for v in self.adj[u]]

    def in_degrees(self):
        deg = {u: 0 for u in self.adj}
        for u in self.adj:
            for v in self.adj[u]:
                deg[v] += 1
        return deg


def kahn_sort(graph: DAG):
    """Topological order by Kahn's algorithm (BFS on in-degrees). Returns the ordering as a
    list, or raises ValueError if the graph has a cycle. Ties are broken by insertion order for
    determinism."""
    indeg = graph.in_degrees()
    order_index = {u: i for i, u in enumerate(graph.adj)}  # stable tie-break
    ready = sorted((u for u in graph.adj if indeg[u] == 0), key=lambda u: order_index[u])
    out = []
    while ready:
        u = ready.pop(0)
        out.append(u)
        newly = []
        for v in graph.adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                newly.append(v)
        # merge newly-ready nodes keeping the stable order
        for v in sorted(newly, key=lambda x: order_index[x]):
            ready.append(v)
        ready.sort(key=lambda x: order_index[x])
    if len(out) != len(graph.adj):
        raise ValueError("graph has a cycle; no topological order exists")
    return out


def dfs_sort(graph: DAG):
    """Topological order by depth-first search (finish-time reversal). Raises ValueError on a
    cycle."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {u: WHITE for u in graph.adj}
    out = []

    def visit(u):
        color[u] = GRAY
        for v in graph.adj[u]:
            if color[v] == GRAY:
                raise ValueError("graph has a cycle; no topological order exists")
            if color[v] == WHITE:
                visit(v)
        color[u] = BLACK
        out.append(u)

    for u in graph.adj:
        if color[u] == WHITE:
            visit(u)
    out.reverse()
    return out


def has_cycle(graph: DAG) -> bool:
    """True if the graph contains a directed cycle."""
    try:
        kahn_sort(graph)
        return False
    except ValueError:
        return True


def is_valid_order(graph: DAG, order) -> bool:
    """True if `order` lists every node exactly once with every edge pointing forward."""
    if sorted(map(id, order)) != sorted(map(id, graph.adj)) and set(order) != set(graph.adj):
        return False
    if len(order) != len(graph.adj):
        return False
    pos = {u: i for i, u in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in graph.edges())


def longest_path(graph: DAG, weights=None):
    """Longest path length through the DAG (the critical path), using unit node/edge weights by
    default or a {node: duration} map. Returns (length, path). Raises on a cycle."""
    order = kahn_sort(graph)  # also validates acyclicity
    w = weights or {u: 1 for u in graph.adj}
    dist = {u: w.get(u, 1) for u in graph.adj}
    pred = {u: None for u in graph.adj}
    for u in order:
        for v in graph.adj[u]:
            if dist[u] + w.get(v, 1) > dist[v]:
                dist[v] = dist[u] + w.get(v, 1)
                pred[v] = u
    end = max(dist, key=lambda u: dist[u])
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = pred[node]
    path.reverse()
    return dist[end], path

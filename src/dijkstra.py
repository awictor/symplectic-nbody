"""Dijkstra's algorithm: shortest paths from one source, greedily.

Given a graph with nonnegative edge weights, what is the cheapest route from a start node to
every other? Edsger Dijkstra's 1959 algorithm answers it by a simple greedy rule: keep a
tentative distance to every node, repeatedly settle the unsettled node with the smallest
tentative distance, and relax its outgoing edges (if going through it is cheaper, lower the
neighbour's distance). Once a node is settled its distance is final -- which works precisely
because edges are nonnegative, so no later, longer detour can ever improve a shorter one.

With a binary-heap priority queue the running time is O((V + E) log V): each edge triggers at
most one heap push, and each pop settles one node. The same relaxation, done V-1 times over all
edges instead of greedily, is the Bellman-Ford algorithm, which is slower but tolerates negative
edges -- a useful cross-check for correctness. Adding a goal-directed heuristic to the priority
turns Dijkstra into A*, the workhorse of game and map routing.

This module builds a weighted graph, runs Dijkstra from a source (returning distances and a
predecessor tree for path reconstruction), reconstructs the actual shortest path to any target,
and includes a from-scratch binary min-heap (no heapq) and a Bellman-Ford implementation used to
verify the distances. Pure stdlib; the shortest-path companion to the Union-Find note (Kruskal
builds trees, Dijkstra walks them)."""

from __future__ import annotations

import math


class MinHeap:
    """A binary min-heap keyed on (priority, item), implemented from scratch (no heapq)."""

    def __init__(self):
        self._h = []

    def __len__(self):
        return len(self._h)

    def push(self, priority, item):
        self._h.append((priority, item))
        self._sift_up(len(self._h) - 1)

    def pop(self):
        """Remove and return the (priority, item) with the smallest priority."""
        if not self._h:
            raise IndexError("pop from empty heap")
        top = self._h[0]
        last = self._h.pop()
        if self._h:
            self._h[0] = last
            self._sift_down(0)
        return top

    def _sift_up(self, i):
        h = self._h
        while i > 0:
            parent = (i - 1) // 2
            if h[i][0] < h[parent][0]:
                h[i], h[parent] = h[parent], h[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        h = self._h
        n = len(h)
        while True:
            left, right = 2 * i + 1, 2 * i + 2
            smallest = i
            if left < n and h[left][0] < h[smallest][0]:
                smallest = left
            if right < n and h[right][0] < h[smallest][0]:
                smallest = right
            if smallest == i:
                break
            h[i], h[smallest] = h[smallest], h[i]
            i = smallest


class Graph:
    """A weighted graph as an adjacency list. Nodes are any hashable; add_edge is directed unless
    `undirected` is set."""

    def __init__(self, undirected: bool = True):
        self.adj = {}
        self.undirected = undirected

    def add_node(self, u):
        self.adj.setdefault(u, [])

    def add_edge(self, u, v, weight: float):
        if weight < 0:
            raise ValueError("Dijkstra requires nonnegative edge weights")
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append((v, weight))
        if self.undirected:
            self.adj[v].append((u, weight))

    def nodes(self):
        return list(self.adj.keys())

    def edges(self):
        """All (u, v, weight) directed edges (each undirected edge appears once per direction)."""
        return [(u, v, w) for u in self.adj for v, w in self.adj[u]]


def dijkstra(graph: Graph, source):
    """Shortest distances from `source` to every reachable node. Returns (dist, prev) where dist
    maps node -> shortest distance (math.inf if unreachable) and prev maps node -> predecessor on
    a shortest path (for reconstruction)."""
    dist = {u: math.inf for u in graph.adj}
    prev = {u: None for u in graph.adj}
    if source not in dist:
        raise KeyError("source not in graph")
    dist[source] = 0.0
    heap = MinHeap()
    heap.push(0.0, source)
    while heap:
        d, u = heap.pop()
        if d > dist[u]:
            continue  # stale entry: we already found a shorter path
        for v, w in graph.adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heap.push(nd, v)
    return dist, prev


def shortest_path(graph: Graph, source, target):
    """The shortest path from source to target as a list of nodes, and its total cost. Returns
    (path, cost); path is empty and cost is inf if target is unreachable."""
    dist, prev = dijkstra(graph, source)
    if dist.get(target, math.inf) == math.inf:
        return [], math.inf
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path, dist[target]


def bellman_ford(graph: Graph, source):
    """Shortest distances by Bellman-Ford (relax all edges V-1 times). Slower than Dijkstra but
    a good independent check; also handles negative edges (though this Graph forbids them)."""
    dist = {u: math.inf for u in graph.adj}
    if source not in dist:
        raise KeyError("source not in graph")
    dist[source] = 0.0
    edges = graph.edges()
    for _ in range(len(dist) - 1):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    return dist

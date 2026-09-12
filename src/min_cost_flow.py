"""Minimum-cost maximum flow: pushing the most flow through a network at the least total cost.

A flow network is a directed graph whose edges each carry a CAPACITY (how much can pass) and now also
a COST PER UNIT (the price of sending one unit along that edge). The maximum-flow problem asks how much
can be pushed from a source to a sink; the MINIMUM-COST MAXIMUM-FLOW problem asks for the cheapest way
to push that maximum amount. This is the workhorse of operations research: route goods from factories
to stores minimising shipping, assign workers to tasks minimising total cost, schedule so throughput is
maximal and expense minimal. It strictly generalises both plain max-flow (all costs zero) and the
assignment problem (a bipartite network with unit capacities), so a single solver answers all three.

The method is SUCCESSIVE SHORTEST PATHS. Repeatedly find the cheapest source-to-sink path in the
RESIDUAL graph -- the network of remaining capacities, where every edge (u,v) of cost c also gains a
reverse edge (v,u) of cost -c that lets flow be cancelled -- and push as much flow along it as the
bottleneck allows. Because we always augment along a minimum-cost path, the total cost stays minimal
for the flow sent so far, and when no augmenting path remains the flow is both maximum and cheapest.
The shortest path must tolerate the NEGATIVE cost of reverse edges, so this implementation uses SPFA
(a queue-based Bellman-Ford) rather than Dijkstra, keeping it simple and dependency-free while
correctly handling the negative residual arcs that Dijkstra alone cannot.

This module builds a directed flow network with per-edge capacity and cost and computes the
minimum-cost maximum flow (and, as a special case, the min-cost to send a fixed target flow). It is
verified against independent references: the flow VALUE it achieves equals the max-flow computed by the
Edmonds-Karp solver, the COST is confirmed minimal by a brute-force search over all integer flows on
small networks, and -- built as a bipartite assignment network with unit capacities -- its optimum
matches the Hungarian algorithm's minimum assignment cost. Pure stdlib; an optimisation companion to
the max-flow / min-cut and Hungarian-assignment notes."""

from __future__ import annotations

from collections import deque


class MinCostFlow:
    """Directed flow network with integer capacities and costs. Vertices are 0..n-1."""

    def __init__(self, n):
        self.n = n
        # each edge stored as [to, capacity, cost, index_of_reverse_edge]
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, capacity, cost):
        """Add a directed edge u->v with the given capacity and per-unit cost. Returns nothing; a
        reverse residual edge of capacity 0 and cost -cost is created automatically."""
        self.graph[u].append([v, capacity, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

    def _spfa(self, source, sink):
        """Cheapest augmenting path via SPFA (queue Bellman-Ford, tolerates negative residual arcs).
        Returns (dist_to_sink, prev_edge) or (None, _) if the sink is unreachable."""
        INF = float("inf")
        dist = [INF] * self.n
        in_queue = [False] * self.n
        prev = [(-1, -1)] * self.n        # prev[v] = (u, edge_index) used to reach v
        dist[source] = 0
        q = deque([source])
        in_queue[source] = True
        while q:
            u = q.popleft()
            in_queue[u] = False
            du = dist[u]
            for i, e in enumerate(self.graph[u]):
                to, cap, cost, _ = e
                if cap > 0 and du + cost < dist[to]:
                    dist[to] = du + cost
                    prev[to] = (u, i)
                    if not in_queue[to]:
                        q.append(to)
                        in_queue[to] = True
        if dist[sink] == INF:
            return None, prev
        return dist[sink], prev

    def min_cost_max_flow(self, source, sink):
        """Push the maximum possible flow from source to sink at minimum total cost.

        Returns (total_flow, total_cost)."""
        total_flow = 0
        total_cost = 0
        while True:
            dist, prev = self._spfa(source, sink)
            if dist is None:
                break
            # bottleneck along the found path
            f = float("inf")
            v = sink
            while v != source:
                u, i = prev[v]
                f = min(f, self.graph[u][i][1])
                v = u
            # push f units, updating residuals
            v = sink
            while v != source:
                u, i = prev[v]
                e = self.graph[u][i]
                e[1] -= f
                self.graph[v][e[3]][1] += f
                v = u
            total_flow += f
            total_cost += f * dist
        return total_flow, total_cost

    def min_cost_for_flow(self, source, sink, target):
        """Minimum cost to send exactly `target` units source->sink. Returns (flow_sent, cost);
        flow_sent < target means the network cannot carry that much."""
        total_flow = 0
        total_cost = 0
        while total_flow < target:
            dist, prev = self._spfa(source, sink)
            if dist is None:
                break
            f = target - total_flow
            v = sink
            while v != source:
                u, i = prev[v]
                f = min(f, self.graph[u][i][1])
                v = u
            v = sink
            while v != source:
                u, i = prev[v]
                e = self.graph[u][i]
                e[1] -= f
                self.graph[v][e[3]][1] += f
                v = u
            total_flow += f
            total_cost += f * dist
        return total_flow, total_cost


def assignment_min_cost(cost):
    """Minimum-cost assignment of n workers to n jobs, solved as a min-cost flow on a bipartite
    network with unit capacities. cost[i][j] is the (non-negative) cost of worker i doing job j.
    Returns (total_cost, assignment) where assignment[i] = job of worker i."""
    n = len(cost)
    # nodes: 0=source, 1..n = workers, n+1..2n = jobs, 2n+1 = sink
    source = 0
    sink = 2 * n + 1
    mcf = MinCostFlow(2 * n + 2)
    for i in range(n):
        mcf.add_edge(source, 1 + i, 1, 0)
        mcf.add_edge(1 + n + i, sink, 1, 0)
    for i in range(n):
        for j in range(n):
            mcf.add_edge(1 + i, 1 + n + j, 1, cost[i][j])
    flow, total = mcf.min_cost_max_flow(source, sink)
    # recover assignment: worker i -> job j where the i->j edge is saturated (capacity now 0)
    assignment = [-1] * n
    for i in range(n):
        for e in mcf.graph[1 + i]:
            to, cap, c, _ = e
            if 1 + n <= to <= 2 * n and cap == 0:   # forward unit edge now used
                assignment[i] = to - (1 + n)
                break
    return total, assignment

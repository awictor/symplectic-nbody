"""Force-directed graph layout: drawing a graph by simulating springs and repulsion.

A graph is a set of nodes and edges with no inherent geometry -- but to SEE its structure you need to
place the nodes on a plane. Force-directed layout treats the drawing as a physical system: every pair
of nodes REPELS like charged particles (so nodes spread out and don't overlap), and every edge acts
like a SPRING pulling its endpoints together (so connected nodes stay near). Let the system relax to
mechanical equilibrium and the low-energy configuration reveals the graph's shape -- clusters bunch,
symmetric graphs draw symmetrically, tree branches fan out. This is how most graph-visualization tools
lay out networks.

The classic FRUCHTERMAN-REINGOLD (1991) algorithm makes the forces scale-free with an ideal edge
length k = C sqrt(area / n):

    repulsive force between every pair:   f_rep(d) = k^2 / d      (pushes apart, all pairs),
    attractive force along every edge:    f_att(d) = d^2 / k      (pulls together, spring-like).

Each iteration sums the forces on every node, moves it along the net force but capped by a COOLING
"temperature" that shrinks over time (simulated annealing), so big rearrangements happen early and
fine adjustments late. The result is a tidy, roughly uniform layout.

This module runs Fruchterman-Reingold to 2-D coordinates, reports the system energy, and offers a
circular initial layout for reproducibility. Validated by the physics and by structure: the total
energy decreases as the layout relaxes; a symmetric graph (a cycle) lays out with nodes at nearly
equal pairwise spacing around a ring; connected nodes end up closer on average than non-connected
ones; two dense clusters joined by one edge separate into two groups; and the layout is deterministic
under a seed. Pure stdlib; the visualization companion to the graph algorithms (BFS, MST, spectral
clustering) throughout the repo."""

from __future__ import annotations

import math


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)

    return nxt


def _circular_init(n, radius=1.0):
    return [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def fruchterman_reingold(n, edges, iterations=200, width=1.0, height=1.0, seed=12345,
                         initial=None):
    """Lay out a graph on [0,width] x [0,height]. edges is a list of (u, v). Returns a list of
    (x, y) node positions. Uses the Fruchterman-Reingold force model with a cooling schedule."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return [(width / 2, height / 2)]
    area = width * height
    k = math.sqrt(area / n)  # ideal edge length

    # adjacency for attractive forces
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].append(v)
            adj[v].append(u)

    rng = _lcg(seed)
    if initial is not None:
        pos = [list(p) for p in initial]
    else:
        pos = [[rng() * width, rng() * height] for _ in range(n)]

    t = width / 10.0  # initial temperature
    cool = t / (iterations + 1)

    for _ in range(iterations):
        disp = [[0.0, 0.0] for _ in range(n)]
        # repulsive forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.hypot(dx, dy) or 1e-9
                force = k * k / dist
                ux, uy = dx / dist, dy / dist
                disp[i][0] += ux * force
                disp[i][1] += uy * force
                disp[j][0] -= ux * force
                disp[j][1] -= uy * force
        # attractive forces along edges
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue  # each edge once
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.hypot(dx, dy) or 1e-9
                force = dist * dist / k
                ux, uy = dx / dist, dy / dist
                disp[i][0] -= ux * force
                disp[i][1] -= uy * force
                disp[j][0] += ux * force
                disp[j][1] += uy * force
        # move nodes, capped by temperature, and keep inside the frame
        for i in range(n):
            dlen = math.hypot(disp[i][0], disp[i][1]) or 1e-9
            step = min(dlen, t)
            pos[i][0] += disp[i][0] / dlen * step
            pos[i][1] += disp[i][1] / dlen * step
            pos[i][0] = min(width, max(0.0, pos[i][0]))
            pos[i][1] = min(height, max(0.0, pos[i][1]))
        t = max(t - cool, 1e-9)

    return [(p[0], p[1]) for p in pos]


def energy(pos, edges, k):
    """The Fruchterman-Reingold potential energy: repulsive -k^2 ln(d) summed over pairs plus
    attractive d^3/(3k) along edges. Lower means a more relaxed layout."""
    n = len(pos)
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]) or 1e-9
            e -= k * k * math.log(d)
    eset = set()
    for u, v in edges:
        if u != v:
            eset.add((min(u, v), max(u, v)))
    for u, v in eset:
        d = math.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]) or 1e-9
        e += d ** 3 / (3 * k)
    return e


def ideal_edge_length(n, width=1.0, height=1.0):
    """k = sqrt(area / n), the target distance for a connected pair."""
    return math.sqrt(width * height / n)


def mean_edge_length(pos, edges):
    """Average Euclidean length of the edges in a layout."""
    eset = set()
    for u, v in edges:
        if u != v:
            eset.add((min(u, v), max(u, v)))
    if not eset:
        return 0.0
    total = sum(math.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]) for u, v in eset)
    return total / len(eset)


def mean_nonedge_length(pos, edges):
    """Average distance between non-adjacent node pairs."""
    n = len(pos)
    eset = set()
    for u, v in edges:
        if u != v:
            eset.add((min(u, v), max(u, v)))
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in eset:
                total += math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
                count += 1
    return total / count if count else 0.0

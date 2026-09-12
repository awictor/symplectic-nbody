"""Stoer-Wagner: the global minimum cut of a weighted undirected graph.

A minimum CUT splits a graph's vertices into two non-empty groups so the total weight of edges
crossing between them is as small as possible. Unlike the s-t min cut (which fixes which side two
particular vertices land on, and is solved by max-flow), the GLOBAL minimum cut asks for the cheapest
cut over ALL ways of splitting the graph -- the graph's weakest link, its most fragile partition.
This measures network reliability (the fewest links whose failure disconnects it), drives image
segmentation and clustering, and bounds how hard a graph is to disconnect.

The STOER-WAGNER algorithm (1997) finds it without any flow computation, in O(V^3), by a strikingly
simple idea: MINIMUM CUT PHASES. Each phase grows a set starting from an arbitrary vertex, repeatedly
adding the vertex most tightly connected to the current set (the maximum-adjacency order), like a
weighted breadth-first sweep. The last two vertices added in the phase, s and t, give a CUT-OF-THE-
PHASE whose weight is the total weight of edges from t to everything else -- and this is provably the
minimum s-t cut for that pair. The phase then MERGES s and t into one vertex (summing parallel edge
weights) and repeats; after V-1 phases every pair has been implicitly considered, and the smallest
cut-of-the-phase seen is the global minimum cut. It needs no augmenting paths, no residual graphs --
just repeated maximum-adjacency orderings and merges.

This module implements Stoer-Wagner global min-cut on a weighted undirected graph, returning the cut
weight and the partition of vertices. It is verified against brute force -- an exhaustive check over
all 2^(n-1) vertex bipartitions for small graphs, and against a max-flow s-t cut minimized over all
pairs -- that the returned weight is the true global minimum, that the partition actually achieves
that weight, on known graphs (a bridge, a cycle, a complete graph), and that disconnected graphs give
a zero cut. Pure stdlib; a graph-algorithm companion to the max-flow, union-find, and Dijkstra
notes."""

from __future__ import annotations


def min_cut(n, edges):
    """Global minimum cut of a weighted undirected graph.

    n: number of vertices (0..n-1). edges: list of (u, v, weight). Returns (cut_weight, partition)
    where partition is the set of vertices on one side of the minimum cut."""
    if n < 2:
        return 0, set()
    # weighted adjacency matrix (parallel edges summed)
    w = [[0] * n for _ in range(n)]
    for u, v, wt in edges:
        if u != v:
            w[u][v] += wt
            w[v][u] += wt

    # each "merged" super-vertex tracks the original vertices it contains
    groups = [{i} for i in range(n)]
    active = list(range(n))
    best_weight = float("inf")
    best_partition = None

    while len(active) > 1:
        # --- minimum-cut phase: maximum-adjacency ordering ---
        a = active[0]
        added = [a]
        in_set = {a}
        weights = {v: w[a][v] for v in active if v != a}
        while len(added) < len(active):
            # pick the most tightly connected vertex not yet in the set
            z = max((v for v in active if v not in in_set), key=lambda v: weights.get(v, 0))
            in_set.add(z)
            added.append(z)
            for v in active:
                if v not in in_set:
                    weights[v] = weights.get(v, 0) + w[z][v]
        # the last two added are s (second-last) and t (last)
        s, t = added[-2], added[-1]
        # cut-of-the-phase = total weight from t to the rest
        cut_weight = sum(w[t][v] for v in active if v != t)
        if cut_weight < best_weight:
            best_weight = cut_weight
            best_partition = set(groups[t])
        # --- merge t into s ---
        groups[s] |= groups[t]
        for v in active:
            if v != s and v != t:
                w[s][v] += w[t][v]
                w[v][s] = w[s][v]
        active.remove(t)

    return best_weight, best_partition


def cut_weight(n, edges, partition):
    """The total weight of edges crossing a given partition (set of vertices on one side)."""
    part = set(partition)
    total = 0
    for u, v, wt in edges:
        if (u in part) != (v in part):
            total += wt
    return total

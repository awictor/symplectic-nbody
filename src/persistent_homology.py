"""Persistent homology (H_0): the barcode of a point cloud, and why its bars are the MST edges.

Ordinary homology counts the holes of ONE space. But real data -- a point cloud sampled from some
shape -- has no single scale: connect points closer than epsilon and you get a different complex for
every epsilon. PERSISTENT HOMOLOGY, the engine of TOPOLOGICAL DATA ANALYSIS, tracks how the topology
changes as epsilon grows from 0 to infinity, recording when each feature is BORN and when it DIES. The
output is a BARCODE (or persistence diagram): long bars are robust features of the underlying shape,
short bars are sampling noise -- a multiscale, coordinate-free summary that is provably stable under
perturbation of the data.

This module computes the zeroth persistent homology, H_0, which tracks CONNECTED COMPONENTS across the
VIETORIS-RIPS filtration (add an edge between two points when their distance drops below the current
epsilon). Every point is born a component at epsilon = 0. As epsilon grows, edges appear in increasing
length order; each edge that joins two DIFFERENT components kills the younger one (the ELDER RULE: the
older component survives). One component never dies -- the whole cloud, its bar runs to infinity.

The beautiful fact this makes concrete: the finite H_0 bars die at EXACTLY the edge weights of the
MINIMUM SPANNING TREE of the point cloud. Persistent H_0 and the MST are the same information. So the
computation is a Kruskal-style union-find sweep over the sorted edges, and it is checked against the
repo's MST: the multiset of finite death times equals the multiset of MST edge lengths, the total
persistence equals the MST weight, and the number of bars alive at any epsilon equals the number of
connected components of the epsilon-graph.

This module builds the pairwise distances, computes the H_0 barcode by the union-find sweep (reusing
the repo's UnionFind), and offers persistence-diagram, Betti-curve, and total-persistence readouts. It
is validated against ground truth: the barcode has exactly n bars with one infinite; the finite deaths
equal the MST edge weights; the Betti_0 curve matches a brute-force component count of the epsilon-graph
at every threshold; well-separated clusters produce exactly that many long bars; and total persistence
equals the MST weight. Pure stdlib; the topological-data-analysis companion to the simplicial-homology,
union-find/MST, and single-linkage clustering tools."""

from __future__ import annotations

import math

from union_find import UnionFind


def pairwise_distances(points):
    """All pairwise Euclidean distances as a sorted list of (dist, i, j) with i < j."""
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(points[i], points[j]) if hasattr(math, "dist") else \
                math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(len(points[i]))))
            edges.append((d, i, j))
    edges.sort()
    return edges


def h0_barcode(points):
    """Zeroth persistent homology of a point cloud over the Vietoris-Rips filtration.

    Returns a list of (birth, death) intervals. Every point is born at 0; a component dies when an
    edge merges it into an older one (elder rule). One bar has death = inf (the whole cloud).
    Also returns the merge edges (dist, i, j) that caused each finite death.
    """
    n = len(points)
    if n == 0:
        return [], []
    edges = pairwise_distances(points)
    uf = UnionFind(n)
    bars = []
    merges = []
    for d, i, j in edges:
        if uf.union(i, j):
            # a merge happened: the younger component dies at epsilon = d
            bars.append((0.0, d))
            merges.append((d, i, j))
            if uf.count() == 1:
                break
    # the surviving component lives forever
    bars.append((0.0, math.inf))
    return bars, merges


def finite_deaths(bars):
    """Sorted list of the finite death times (the lengths of the MST edges)."""
    return sorted(b[1] for b in bars if b[1] != math.inf)


def total_persistence(bars):
    """Sum of the finite bar lengths = the minimum-spanning-tree weight."""
    return sum(b[1] - b[0] for b in bars if b[1] != math.inf)


def betti0_curve(bars, epsilons):
    """Number of H_0 bars alive at each epsilon (the connected-component count of the eps-graph)."""
    out = []
    for eps in epsilons:
        alive = sum(1 for birth, death in bars if birth <= eps < death)
        out.append(alive)
    return out


def brute_components(points, eps):
    """Reference: number of connected components of the graph joining points within distance eps."""
    n = len(points)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(len(points[i]))))
            if d <= eps:
                uf.union(i, j)
    return uf.count()


def mst_edge_weights(points):
    """Reference: the sorted edge weights of the Euclidean minimum spanning tree (Kruskal)."""
    n = len(points)
    edges = pairwise_distances(points)
    uf = UnionFind(n)
    weights = []
    for d, i, j in edges:
        if uf.union(i, j):
            weights.append(d)
            if len(weights) == n - 1:
                break
    return sorted(weights)


def persistence_diagram(bars, inf_value=None):
    """The (birth, death) points for plotting; inf deaths mapped to inf_value if given."""
    pts = []
    for birth, death in bars:
        if death == math.inf and inf_value is not None:
            death = inf_value
        pts.append((birth, death))
    return pts

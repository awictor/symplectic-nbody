"""Boruvka's algorithm: the oldest minimum spanning tree method, and the most parallel.

Boruvka found this in 1926 -- decades before Kruskal or Prim -- while planning the electrical grid of
Moravia, making it arguably the first graph algorithm ever published. It builds the MINIMUM SPANNING
TREE, the cheapest set of edges connecting every vertex, but by a strategy neither of the famous later
algorithms uses, and one that maps beautifully onto parallel hardware.

The idea is component growth in synchronized rounds. Start with every vertex its own component. In each
round, EVERY component simultaneously finds its own CHEAPEST OUTGOING EDGE -- the lightest edge leaving
it to any other component -- and all those edges are added to the tree at once, merging the components
they join. Because at least half the components disappear each round (each surviving component absorbs
at least one other), only O(log V) rounds are needed, and each round scans the edges in O(E). The total
is O(E log V), the same as Kruskal, but the per-round work is embarrassingly parallel: the cheapest-edge
search for different components is independent, which is why Boruvka underlies modern parallel and GPU
MST implementations.

The one subtlety is CORRECTNESS UNDER TIES. If two components each pick an edge of equal weight, naive
merging can form a cycle (both pick the same edge from opposite ends, or a 3-cycle of equal edges). The
standard fix is a deterministic tie-break: compare equal-weight edges by an index, so a consistent
global order breaks ties and no cycle can form. This is the cut property in action -- the cheapest edge
crossing any partition is safe -- applied to all components at once.

This module implements Boruvka's algorithm with union-find components and the index tie-break, returning
the MST weight and edges. It is validated against Kruskal's algorithm from the repo's union-find module:
the two produce the same total weight on random graphs (the MST weight is unique even when the edge set
is not); the result is always a spanning tree (V-1 edges, connected, acyclic) on connected graphs and a
spanning forest otherwise; it handles equal weights without forming cycles; and it matches a brute-force
minimum over all spanning trees on tiny graphs. Pure stdlib; the round-based, parallel-friendly
companion to the Kruskal and Prim MST tools and the union-find structure."""

from __future__ import annotations

from union_find import UnionFind, kruskal_mst


def boruvka_mst(n, weighted_edges):
    """Minimum spanning tree (or forest) by Boruvka's algorithm.

    `weighted_edges` is a list of (weight, u, v). Returns (total_weight, chosen_edges) where
    chosen_edges is a list of (weight, u, v). Edges are added in the rounds Boruvka discovers them.
    """
    uf = UnionFind(n)
    total = 0.0
    chosen = []
    # index each edge for a deterministic tie-break
    edges = [(w, u, v, idx) for idx, (w, u, v) in enumerate(weighted_edges)]

    num_components = n
    while num_components > 1:
        # cheapest outgoing edge for each component root
        cheapest = {}  # root -> (weight, index, edge_tuple)
        for w, u, v, idx in edges:
            ru, rv = uf.find(u), uf.find(v)
            if ru == rv:
                continue  # internal edge, skip
            key = (w, idx)
            for r in (ru, rv):
                if r not in cheapest or key < cheapest[r][0]:
                    cheapest[r] = (key, (w, u, v))

        if not cheapest:
            break  # graph is disconnected; no more cross-component edges

        added_any = False
        for r, (key, (w, u, v)) in cheapest.items():
            if uf.union(u, v):   # only if it actually merges two components
                total += w
                chosen.append((w, u, v))
                added_any = True
        if not added_any:
            break

        # recount components
        num_components = len({uf.find(i) for i in range(n)})

    return total, chosen


def is_spanning_forest(n, chosen, weighted_edges):
    """Check `chosen` is a spanning forest: acyclic and connects the same components as the full graph."""
    uf_full = UnionFind(n)
    for w, u, v in weighted_edges:
        uf_full.union(u, v)
    full_components = len({uf_full.find(i) for i in range(n)})

    uf = UnionFind(n)
    for w, u, v in chosen:
        if not uf.union(u, v):
            return False  # a cycle -> not a forest
    tree_components = len({uf.find(i) for i in range(n)})
    # a spanning forest has exactly (n - full_components) edges
    return tree_components == full_components and len(chosen) == n - full_components


def brute_mst_weight(n, weighted_edges):
    """Brute-force minimum spanning-tree weight by trying all edge subsets of size n-1 (tiny graphs)."""
    from itertools import combinations
    m = len(weighted_edges)
    if n == 1:
        return 0.0
    best = None
    for combo in combinations(range(m), n - 1):
        uf = UnionFind(n)
        ok = True
        total = 0.0
        for idx in combo:
            w, u, v = weighted_edges[idx]
            if not uf.union(u, v):
                ok = False
                break
            total += w
        if ok and len({uf.find(i) for i in range(n)}) == 1:
            if best is None or total < best:
                best = total
    return best


def kruskal_weight(n, weighted_edges):
    """Convenience wrapper: MST weight via the repo's Kruskal implementation (for cross-checks)."""
    return kruskal_mst(n, weighted_edges)[0]

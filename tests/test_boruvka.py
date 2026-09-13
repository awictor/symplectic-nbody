"""Tests for Boruvka MST: matches Kruskal weight, spanning forest, ties, brute force on tiny graphs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from boruvka import (  # noqa: E402
    boruvka_mst,
    is_spanning_forest,
    brute_mst_weight,
    kruskal_weight,
)
from union_find import UnionFind  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _random_connected_graph(n, extra_edges, rng, wmax=100):
    """A random connected graph: a spanning path plus extra random edges."""
    edges = []
    for i in range(1, n):
        w = 1 + int(rng() * wmax)
        edges.append((w, i - 1, i))
    for _ in range(extra_edges):
        u = int(rng() * n)
        v = int(rng() * n)
        if u != v:
            edges.append((1 + int(rng() * wmax), u, v))
    return edges


def main():
    # ---- 1. matches Kruskal weight on random connected graphs ---------------------------
    rng = _lcg(1)
    ok = True
    for trial in range(30):
        n = 5 + int(rng() * 15)
        edges = _random_connected_graph(n, n, rng)
        bw, _ = boruvka_mst(n, edges)
        kw = kruskal_weight(n, edges)
        if abs(bw - kw) > 1e-9:
            ok = False
            check("Boruvka weight == Kruskal", False, f"n={n}: {bw} vs {kw}")
            break
    if ok:
        check("Boruvka weight == Kruskal (30 random graphs)", True)

    # ---- 2. result is a spanning tree (V-1 edges, connected) ----------------------------
    rng = _lcg(2)
    n = 12
    edges = _random_connected_graph(n, 20, rng)
    w, chosen = boruvka_mst(n, edges)
    check("spanning tree has n-1 edges", len(chosen) == n - 1, f"{len(chosen)}")
    check("result is a spanning forest", is_spanning_forest(n, chosen, edges))
    uf = UnionFind(n)
    for cw, u, v in chosen:
        uf.union(u, v)
    check("MST connects all vertices", uf.count() == 1, f"{uf.count()} components")

    # ---- 3. MST is acyclic (union never rejects a chosen edge) --------------------------
    uf = UnionFind(n)
    acyclic = all(uf.union(u, v) for cw, u, v in chosen)
    check("MST is acyclic", acyclic)

    # ---- 4. matches brute-force minimum over all spanning trees (tiny graphs) -----------
    rng = _lcg(3)
    ok = True
    for trial in range(15):
        n = 4 + int(rng() * 3)  # 4..6 vertices
        edges = _random_connected_graph(n, n, rng)
        bw, _ = boruvka_mst(n, edges)
        brute = brute_mst_weight(n, edges)
        if abs(bw - brute) > 1e-9:
            ok = False
            check("Boruvka == brute force", False, f"n={n}: {bw} vs {brute}")
            break
    if ok:
        check("Boruvka == brute-force MST weight (15 tiny graphs)", True)

    # ---- 5. equal weights don't create cycles -------------------------------------------
    # a graph where every edge has the same weight
    n = 8
    uniform = []
    for i in range(1, n):
        uniform.append((5, i - 1, i))
    uniform.append((5, 0, 7))
    uniform.append((5, 3, 6))
    uniform.append((5, 2, 5))
    w, chosen = boruvka_mst(n, uniform)
    uf = UnionFind(n)
    no_cycle = all(uf.union(u, v) for cw, u, v in chosen)
    check("equal weights -> no cycle", no_cycle and len(chosen) == n - 1, f"{len(chosen)} edges")
    check("equal-weight MST weight = 5*(n-1)", abs(w - 5 * (n - 1)) < 1e-9, f"{w}")

    # ---- 6. disconnected graph -> spanning forest ---------------------------------------
    # two separate triangles
    forest_edges = [(1, 0, 1), (2, 1, 2), (3, 0, 2),   # component {0,1,2}
                    (1, 3, 4), (2, 4, 5), (5, 3, 5)]   # component {3,4,5}
    w, chosen = boruvka_mst(6, forest_edges)
    check("forest has (n - components) edges", len(chosen) == 6 - 2, f"{len(chosen)}")
    check("forest is a valid spanning forest", is_spanning_forest(6, chosen, forest_edges))
    # each triangle contributes its two lightest edges: (1+2) + (1+2) = 6
    check("forest weight = 6", abs(w - 6) < 1e-9, f"{w}")

    # ---- 7. single vertex, single edge, two vertices ------------------------------------
    check("single vertex MST weight 0", boruvka_mst(1, [])[0] == 0.0)
    check("two vertices one edge", boruvka_mst(2, [(7, 0, 1)]) == (7, [(7, 0, 1)]))

    # ---- 8. parallel edges (multigraph): cheapest is chosen -----------------------------
    multi = [(10, 0, 1), (3, 0, 1), (7, 0, 1)]
    w, chosen = boruvka_mst(2, multi)
    check("parallel edges -> cheapest", abs(w - 3) < 1e-9 and len(chosen) == 1, f"{w}")

    # ---- 9. weight is unique even if the edge set differs from Kruskal's ----------------
    rng = _lcg(9)
    n = 10
    edges = _random_connected_graph(n, 25, rng)
    bw, bchosen = boruvka_mst(n, edges)
    kw = kruskal_weight(n, edges)
    check("unique MST weight across algorithms", abs(bw - kw) < 1e-9, f"{bw} vs {kw}")

    # ---- 10. larger graph terminates and is a tree --------------------------------------
    rng = _lcg(10)
    n = 60
    edges = _random_connected_graph(n, 150, rng)
    w, chosen = boruvka_mst(n, edges)
    check("large graph -> spanning tree", len(chosen) == n - 1 and is_spanning_forest(n, chosen, edges))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for Gomory-Hu: tree path-min equals brute-force s-t min cut for every pair."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gomory_hu import (  # noqa: E402
    gomory_hu_tree,
    min_cut_st,
    min_cut_query,
    all_pairs_min_cuts,
)


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
        return state >> 8

    return nxt


def _random_graph(n, rng, max_w=9, edge_prob=0.6):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng() % 100 < edge_prob * 100:
                w = 1 + rng() % max_w
                edges.append((u, v, w))
    return edges


def main():
    # ---- 1. tree matches brute-force min cut for EVERY pair, many graphs ----------------
    rng = _lcg(2024)
    mismatches = 0
    total_pairs = 0
    disconnected_ok = 0
    for trial in range(200):
        n = 2 + rng() % 6  # 2..7 vertices
        edges = _random_graph(n, rng)
        tree = gomory_hu_tree(n, edges)
        for s in range(n):
            for t in range(s + 1, n):
                brute, _ = min_cut_st(n, edges, s, t)
                tree_val = min_cut_query(n, tree, s, t)
                total_pairs += 1
                if brute != tree_val:
                    mismatches += 1
                if brute == 0:
                    disconnected_ok += 1
    check("tree path-min == brute min cut for all pairs (200 graphs)",
          mismatches == 0, f"{mismatches}/{total_pairs} mismatched")
    check("test corpus exercised disconnected pairs (cut 0)", disconnected_ok > 0,
          f"{disconnected_ok} zero-cut pairs seen")

    # ---- 2. tree has exactly n-1 edges --------------------------------------------------
    rng = _lcg(77)
    ok = True
    for _ in range(30):
        n = 3 + rng() % 5
        edges = _random_graph(n, rng, edge_prob=0.8)
        tree = gomory_hu_tree(n, edges)
        if len(tree) != n - 1:
            ok = False
    check("tree has exactly n-1 edges", ok)

    # ---- 3. all_pairs table symmetric and matches per-query -----------------------------
    rng = _lcg(555)
    n = 6
    edges = _random_graph(n, rng, edge_prob=0.9)
    tree = gomory_hu_tree(n, edges)
    table = all_pairs_min_cuts(n, tree)
    sym = all(table[i][j] == table[j][i] for i in range(n) for j in range(n))
    diag = all(table[i][i] == 0 for i in range(n))
    matches_query = all(
        table[i][j] == min_cut_query(n, tree, i, j)
        for i in range(n) for j in range(i + 1, n)
    )
    check("all-pairs table symmetric", sym)
    check("all-pairs table zero diagonal", diag)
    check("all-pairs table matches per-pair query", matches_query)

    # ---- 4. hand example: a path graph 0-1-2-3 with weights ------------------------------
    # edges: (0,1,3), (1,2,1), (2,3,5). Min cut between 0 and 3 is the bottleneck edge (1,2,1).
    edges = [(0, 1, 3), (1, 2, 1), (2, 3, 5)]
    tree = gomory_hu_tree(4, edges)
    check("path graph: cut(0,3) = bottleneck 1", min_cut_query(4, tree, 0, 3) == 1)
    check("path graph: cut(0,1) = 3", min_cut_query(4, tree, 0, 1) == 3)
    check("path graph: cut(2,3) = 5", min_cut_query(4, tree, 2, 3) == 5)

    # ---- 5. hand example: a 4-cycle, every min cut is 2 (two edges of weight 1) ---------
    edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]
    tree = gomory_hu_tree(4, edges)
    all_two = all(min_cut_query(4, tree, s, t) == 2 for s in range(4) for t in range(s + 1, 4))
    check("4-cycle: every pair min cut = 2", all_two)

    # ---- 6. complete graph K4 unit weights: every min cut = degree 3 ---------------------
    edges = [(u, v, 1) for u in range(4) for v in range(u + 1, 4)]
    tree = gomory_hu_tree(4, edges)
    all_three = all(min_cut_query(4, tree, s, t) == 3 for s in range(4) for t in range(s + 1, 4))
    check("K4 unit: every pair min cut = degree 3", all_three)

    # ---- 7. bridge dumbbell: cut across the bridge = bridge weight ----------------------
    # two triangles {0,1,2} and {3,4,5} joined by a single bridge (2,3,w=2)
    edges = [
        (0, 1, 5), (1, 2, 5), (0, 2, 5),
        (3, 4, 5), (4, 5, 5), (3, 5, 5),
        (2, 3, 2),
    ]
    tree = gomory_hu_tree(6, edges)
    check("dumbbell: cut across bridge = 2", min_cut_query(6, tree, 0, 5) == 2)
    check("dumbbell: cut within a triangle = 10", min_cut_query(6, tree, 0, 1) == 10)

    # ---- 8. single vertex / empty edge cases --------------------------------------------
    check("n=1 tree empty", gomory_hu_tree(1, []) == [])
    check("n=2 no edges: cut 0", min_cut_query(2, gomory_hu_tree(2, []), 0, 1) == 0)
    check("n=2 one edge w=4: cut 4",
          min_cut_query(2, gomory_hu_tree(2, [(0, 1, 4)]), 0, 1) == 4)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

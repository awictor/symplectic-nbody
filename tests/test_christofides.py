"""Tests for Christofides: <=1.5x Held-Karp optimum, valid tour, Eulerian multigraph, min matching."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from christofides import (  # noqa: E402
    christofides,
    euclidean_matrix,
    tour_length,
    _mst_edges,
    odd_degree_vertices,
    min_weight_perfect_matching,
    brute_min_matching,
    eulerian_circuit,
)
from tsp import held_karp  # noqa: E402


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


def main():
    # ---- 1. tour is a valid permutation of all cities -----------------------------------
    rng = _lcg(1)
    pts = [(rng() * 100, rng() * 100) for _ in range(9)]
    D = euclidean_matrix(pts)
    tour, length = christofides(D)
    check("tour visits every city once", sorted(tour) == list(range(9)), f"{sorted(tour)}")

    # ---- 2. returned length matches independent recomputation ---------------------------
    check("length matches recomputed tour_length", abs(length - tour_length(tour, D)) < 1e-9)

    # ---- 3. never exceeds 1.5x the exact Held-Karp optimum (many random metric instances)
    worst_ratio = 0.0
    for seed in range(20):
        r = _lcg(seed + 100)
        n = 8
        p = [(r() * 100, r() * 100) for _ in range(n)]
        Dm = euclidean_matrix(p)
        _, opt = held_karp(Dm)
        _, approx = christofides(Dm)
        ratio = approx / opt if opt > 0 else 1.0
        worst_ratio = max(worst_ratio, ratio)
        if ratio > 1.5 + 1e-9:
            check(f"seed {seed}: <=1.5x optimum", False, f"ratio {ratio:.4f}")
            break
    else:
        check(f"<=1.5x optimum over 20 instances (worst {worst_ratio:.4f})", True)

    # ---- 4. MST has exactly n-1 edges and an even number of odd-degree vertices ---------
    tree = _mst_edges(9, D)
    check("MST has n-1 edges", len(tree) == 8, f"{len(tree)}")
    odds = odd_degree_vertices(9, tree)
    check("odd-degree vertex count is even", len(odds) % 2 == 0, f"{len(odds)}")

    # ---- 5. tree + matching gives an all-even-degree (Eulerian) multigraph --------------
    match_edges, _ = min_weight_perfect_matching(odds, D)
    multi = list(tree) + list(match_edges)
    deg = [0] * 9
    for u, v in multi:
        deg[u] += 1
        deg[v] += 1
    check("tree+matching all even degree", all(d % 2 == 0 for d in deg), f"{deg}")

    # ---- 6. Eulerian circuit uses every multigraph edge exactly once --------------------
    circ = eulerian_circuit(9, multi, start=0)
    # a closed circuit of E edges visits E+1 vertices in sequence
    check("Eulerian circuit length = edges+1", len(circ) == len(multi) + 1,
          f"{len(circ)} vs {len(multi)+1}")
    check("Eulerian circuit is closed", circ[0] == circ[-1], f"{circ[0]} {circ[-1]}")

    # ---- 7. exact matching agrees with brute force --------------------------------------
    for seed in range(6):
        r = _lcg(seed + 500)
        p = [(r() * 50, r() * 50) for _ in range(6)]
        Dm = euclidean_matrix(p)
        verts = [0, 1, 2, 3, 4, 5]
        _, cost = min_weight_perfect_matching(verts, Dm)
        bcost = brute_min_matching(verts, Dm)
        if abs(cost - bcost) > 1e-9:
            check("matching == brute force", False, f"{cost} vs {bcost}")
            break
    else:
        check("min-weight matching == brute force (6 cases)", True)

    # ---- 8. degenerate sizes ------------------------------------------------------------
    check("n=1 trivial", christofides([[0.0]]) == ([0], 0.0))
    t2, l2 = christofides([[0.0, 3.0], [3.0, 0.0]])
    check("n=2 tour", sorted(t2) == [0, 1] and abs(l2 - 6.0) < 1e-9, f"{t2} {l2}")
    check("n=0 empty", christofides([]) == ([], 0.0))

    # ---- 9. matching parity guard -------------------------------------------------------
    try:
        min_weight_perfect_matching([0, 1, 2], D)
        check("odd matching size raises", False)
    except ValueError:
        check("odd matching size raises", True)

    # ---- 10. collinear metric instance: optimal is the sorted pass, approx within bound -
    line_pts = [(x, 0.0) for x in [0, 5, 2, 9, 3, 7]]
    Dl = euclidean_matrix(line_pts)
    _, optl = held_karp(Dl)
    _, appl = christofides(Dl)
    check("collinear within 1.5x", appl <= 1.5 * optl + 1e-9, f"{appl} vs opt {optl}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

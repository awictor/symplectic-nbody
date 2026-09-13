"""Tests for force-directed layout: energy decreases, symmetry, edges closer, clusters separate."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from force_layout import (  # noqa: E402
    fruchterman_reingold,
    energy,
    ideal_edge_length,
    mean_edge_length,
    mean_nonedge_length,
    _circular_init,
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


def main():
    # ---- 1. energy decreases as the layout relaxes --------------------------------------
    # a random-ish graph; compare energy of the initial vs relaxed layout
    n = 12
    edges = [(i, (i + 1) % n) for i in range(n)] + [(0, 6), (3, 9)]
    k = ideal_edge_length(n)
    init = _circular_init(n, radius=0.05)  # start bunched near center-ish
    init = [(0.5 + p[0], 0.5 + p[1]) for p in init]
    e0 = energy(init, edges, k)
    pos = fruchterman_reingold(n, edges, iterations=300, seed=7, initial=init)
    e1 = energy(pos, edges, k)
    check("energy decreases after relaxation", e1 < e0, f"{e0:.2f} -> {e1:.2f}")

    # ---- 2. connected nodes end up closer than non-connected ones -----------------------
    n = 15
    # a graph with clear structure: a path
    edges = [(i, i + 1) for i in range(n - 1)]
    pos = fruchterman_reingold(n, edges, iterations=400, seed=3)
    me = mean_edge_length(pos, edges)
    mn = mean_nonedge_length(pos, edges)
    check("edges shorter than non-edges on average", me < mn, f"edge {me:.3f} vs non {mn:.3f}")

    # ---- 3. cycle graph lays out with roughly equal pairwise spacing --------------------
    n = 10
    edges = [(i, (i + 1) % n) for i in range(n)]
    pos = fruchterman_reingold(n, edges, iterations=500, seed=5)
    # neighbouring edge lengths should be nearly equal (symmetry)
    lengths = [math.hypot(pos[i][0] - pos[(i + 1) % n][0], pos[i][1] - pos[(i + 1) % n][1])
               for i in range(n)]
    mean_l = sum(lengths) / n
    spread = max(abs(l - mean_l) for l in lengths) / mean_l
    check("cycle: edge lengths nearly equal (symmetric ring)", spread < 0.25,
          f"relative spread {spread:.3f}")

    # ---- 4. two clusters joined by one edge separate ------------------------------------
    # cluster A = 0..4 (complete), cluster B = 5..9 (complete), bridge (4,5)
    edges = []
    for i in range(5):
        for j in range(i + 1, 5):
            edges.append((i, j))
    for i in range(5, 10):
        for j in range(i + 1, 10):
            edges.append((i, j))
    edges.append((4, 5))
    pos = fruchterman_reingold(10, edges, iterations=500, seed=11)
    # centroid of each cluster
    def centroid(idxs):
        return (sum(pos[i][0] for i in idxs) / len(idxs),
                sum(pos[i][1] for i in idxs) / len(idxs))
    cA = centroid(range(5))
    cB = centroid(range(5, 10))
    inter = math.hypot(cA[0] - cB[0], cA[1] - cB[1])
    # average intra-cluster spread
    def spread(idxs, c):
        return sum(math.hypot(pos[i][0] - c[0], pos[i][1] - c[1]) for i in idxs) / len(idxs)
    intra = (spread(range(5), cA) + spread(range(5, 10), cB)) / 2
    check("two clusters separate (inter-centroid > intra-spread)", inter > intra,
          f"inter {inter:.3f} vs intra {intra:.3f}")

    # ---- 5. determinism under seed ------------------------------------------------------
    p1 = fruchterman_reingold(8, [(0, 1), (1, 2), (2, 3)], iterations=100, seed=42)
    p2 = fruchterman_reingold(8, [(0, 1), (1, 2), (2, 3)], iterations=100, seed=42)
    check("deterministic under seed", p1 == p2)
    p3 = fruchterman_reingold(8, [(0, 1), (1, 2), (2, 3)], iterations=100, seed=99)
    check("different seeds differ", p1 != p3)

    # ---- 6. all nodes stay within the frame ---------------------------------------------
    pos = fruchterman_reingold(20, [(i, (i * 7) % 20) for i in range(20)],
                               iterations=200, width=2.0, height=3.0, seed=1)
    check("nodes stay within the frame", all(0 <= x <= 2.0 and 0 <= y <= 3.0 for x, y in pos))

    # ---- 7. edge cases ------------------------------------------------------------------
    check("single node centered", fruchterman_reingold(1, [], width=2, height=2) == [(1.0, 1.0)])
    check("two nodes produced", len(fruchterman_reingold(2, [(0, 1)])) == 2)
    check("no edges (all repulsion) spreads out",
          mean_nonedge_length(fruchterman_reingold(6, [], iterations=200, seed=2), []) > 0)
    try:
        fruchterman_reingold(0, [])
        check("n=0 raises", False)
    except ValueError:
        check("n=0 raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for stoer_wagner: global min cut vs brute force, known graphs, partition validity."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stoer_wagner import min_cut, cut_weight

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 555
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_min_cut(n, edges):
    best = float("inf")
    for r in range(1, n):
        for combo in itertools.combinations(range(n), r):
            best = min(best, cut_weight(n, edges, set(combo)))
    return best


# --- the classic Stoer-Wagner textbook graph (global min cut = 4) ----------
classic = [(0, 1, 2), (0, 4, 3), (1, 2, 3), (1, 4, 2), (1, 5, 2), (2, 3, 4),
           (2, 6, 2), (3, 6, 2), (3, 7, 2), (4, 5, 3), (5, 6, 1), (6, 7, 3)]
cw, part = min_cut(8, classic)
check("classic graph global min cut is 4", cw == 4)
check("returned partition achieves the cut weight", cut_weight(8, classic, part) == cw)
check("partition is non-empty and not everything", 0 < len(part) < 8)

# --- matches brute force over random small graphs --------------------------
ok = True
for _ in range(60):
    n = 3 + int(rng() * 5)          # 3..7 vertices
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng() < 0.6:
                edges.append((u, v, 1 + int(rng() * 9)))
    if len(edges) < n - 1:
        continue                    # skip likely-disconnected sparse graphs for the weight check
    cw, part = min_cut(n, edges)
    bf = brute_min_cut(n, edges)
    if cw != bf:
        ok = False
        break
    # the partition must actually achieve the weight
    if cut_weight(n, edges, part) != cw:
        ok = False
        break
check("global min cut matches brute force over random graphs", ok)

# --- a bridge: the min cut is the single bridging edge ---------------------
bridge = [(0, 1, 5), (1, 2, 5), (0, 2, 5), (2, 3, 1), (3, 4, 5), (4, 5, 5), (3, 5, 5)]
cw, part = min_cut(6, bridge)
check("bridge graph min cut is the bridge weight (1)", cw == 1)
check("bridge partition splits the two triangles",
      part == {0, 1, 2} or part == {3, 4, 5})

# --- a cycle: min cut is 2 (must cut two edges to split a ring) ------------
cycle = [(i, (i + 1) % 6, 5) for i in range(6)]
cw, _ = min_cut(6, cycle)
check("uniform cycle min cut is 2 edges * weight (10)", cw == 10)

# --- a complete graph K_n with unit weights: min cut is n-1 ----------------
# isolating one vertex cuts its n-1 edges, which is optimal for uniform K_n
for n in [4, 5, 6]:
    kn = [(u, v, 1) for u in range(n) for v in range(u + 1, n)]
    cw, part = min_cut(n, kn)
    check(f"K_{n} unit-weight global min cut is {n-1}", cw == n - 1)

# --- a weighted path graph -------------------------------------------------
# 0-1-2-3-4 with weights; min cut is the lightest single edge
path = [(0, 1, 4), (1, 2, 2), (2, 3, 7), (3, 4, 5)]
cw, part = min_cut(5, path)
check("weighted path min cut is the lightest edge (2)", cw == 2)

# --- two disconnected components: min cut is 0 -----------------------------
disc = [(0, 1, 3), (1, 2, 3), (3, 4, 3)]     # {0,1,2} and {3,4} disconnected
cw, part = min_cut(5, disc)
check("disconnected graph has a zero min cut", cw == 0)

# --- two vertices, one edge ------------------------------------------------
cw, part = min_cut(2, [(0, 1, 7)])
check("two vertices: min cut is the single edge weight", cw == 7)

# --- symmetry: the partition and its complement give the same cut ----------
cw, part = min_cut(8, classic)
complement = set(range(8)) - part
check("partition and complement give the same cut weight",
      cut_weight(8, classic, part) == cut_weight(8, classic, complement))

# --- parallel edges are summed ---------------------------------------------
parallel = [(0, 1, 2), (0, 1, 3), (1, 2, 4)]     # 0-1 has total weight 5
cw, _ = min_cut(3, parallel)
check("parallel edges summed: min cut is 4 (isolate vertex 2)", cw == 4)

# --- larger random graph: partition always achieves the reported weight ----
ok = True
for _ in range(20):
    n = 10 + int(rng() * 10)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng() < 0.4:
                edges.append((u, v, 1 + int(rng() * 5)))
    cw, part = min_cut(n, edges)
    if part is not None and cut_weight(n, edges, part) != cw:
        ok = False
        break
check("reported partition always achieves the reported cut weight (large graphs)", ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all stoer_wagner tests passed")

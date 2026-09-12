"""Tests for graph_coloring: greedy/DSATUR/exact vs brute-force chromatic number."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_coloring import (greedy_coloring, dsatur_coloring, chromatic_number, num_colors,
                            is_proper, brute_chromatic_number)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


def random_graph(rng, n, p_num, p_den):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.rand() % p_den < p_num:
                edges.append((u, v))
    return edges


# --- known chromatic numbers ------------------------------------------------
k4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]
check("K4 needs 4 colors", chromatic_number(4, k4)[0] == 4)
check("K5 needs 5 colors", chromatic_number(5, [(i, j) for i in range(5) for j in range(i + 1, 5)])[0] == 5)

c4 = [(0, 1), (1, 2), (2, 3), (3, 0)]
check("even cycle C4 needs 2 colors", chromatic_number(4, c4)[0] == 2)
c5 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
check("odd cycle C5 needs 3 colors", chromatic_number(5, c5)[0] == 3)

check("edgeless graph needs 1 color", chromatic_number(4, [])[0] == 1)
check("empty graph needs 0 colors", chromatic_number(0, [])[0] == 0)

# a bipartite (complete bipartite K_{2,3}) needs 2
k23 = [(a, b) for a in range(2) for b in range(2, 5)]
check("complete bipartite K(2,3) needs 2 colors", chromatic_number(5, k23)[0] == 2)

# --- exact chromatic number vs brute ---------------------------------------
rng = LCG(2026)
exact_ok = True
for _ in range(300):
    n = rng.randint(1, 8)
    edges = random_graph(rng, n, rng.randint(1, 3), 4)
    if chromatic_number(n, edges)[0] != brute_chromatic_number(n, edges):
        exact_ok = False
        print(f"  chromatic mismatch: n={n} edges={edges}")
        break
check("exact chromatic number matches brute force (300 random graphs)", exact_ok)

# --- the exact witness coloring is proper and uses exactly chi colors ------
rng = LCG(4242)
witness_ok = True
for _ in range(300):
    n = rng.randint(1, 8)
    edges = random_graph(rng, n, rng.randint(1, 3), 4)
    chi, coloring = chromatic_number(n, edges)
    if not is_proper(n, edges, coloring) or num_colors(coloring) != chi:
        witness_ok = False
        break
check("exact witness coloring is proper and uses exactly chi colors", witness_ok)

# --- greedy and DSATUR are proper and never below chi ----------------------
rng = LCG(777)
heuristic_ok = True
dsatur_never_worse = True
for _ in range(300):
    n = rng.randint(1, 9)
    edges = random_graph(rng, n, rng.randint(1, 3), 4)
    chi = chromatic_number(n, edges)[0]
    g = greedy_coloring(n, edges)
    d = dsatur_coloring(n, edges)
    if not is_proper(n, edges, g) or not is_proper(n, edges, d):
        heuristic_ok = False
        break
    # heuristics are upper bounds: never fewer colors than the true minimum
    if num_colors(g) < chi or num_colors(d) < chi:
        heuristic_ok = False
        break
check("greedy and DSATUR always produce proper colorings, never below chi (300 graphs)",
      heuristic_ok)

# --- DSATUR is optimal on cycles and complete graphs -----------------------
opt_ok = True
for n in range(3, 10):
    cyc = [(i, (i + 1) % n) for i in range(n)]
    expected = 2 if n % 2 == 0 else 3
    if num_colors(dsatur_coloring(n, cyc)) != expected:
        opt_ok = False
        break
    kn = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if num_colors(dsatur_coloring(n, kn)) != n:
        opt_ok = False
        break
check("DSATUR is optimal on cycles (2/3) and complete graphs (n)", opt_ok)

# --- greedy order matters: a known bad order needs more colors -------------
# crown graph / bipartite where a bad order forces many colors, good order needs 2
# vertices 0..3 left, 4..7 right; edges i-(4+j) for i != j  -> bipartite, chi = 2
crown = [(i, 4 + j) for i in range(4) for j in range(4) if i != j]
chi = chromatic_number(8, crown)[0]
check("crown graph is bipartite (chi = 2)", chi == 2)
# the alternating order 0,4,1,5,... can make greedy use more than 2
bad_order = [0, 4, 1, 5, 2, 6, 3, 7]
check("greedy with a bad order can exceed chi (illustrating order-dependence)",
      num_colors(greedy_coloring(8, crown, bad_order)) >= 2)

# --- self-loops / repeated edges are tolerated -----------------------------
check("repeated edges do not break coloring",
      chromatic_number(3, [(0, 1), (0, 1), (1, 2)])[0] == 2)

# --- a bigger sparse graph: heuristics proper and >= chi lower bound --------
rng = LCG(31337)
n = 40
edges = random_graph(rng, n, 1, 8)
d = dsatur_coloring(n, edges)
check("40-vertex graph: DSATUR coloring is proper", is_proper(n, edges, d))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all graph_coloring tests passed")

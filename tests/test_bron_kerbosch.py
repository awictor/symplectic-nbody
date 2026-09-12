"""Tests for bron_kerbosch: maximal-clique enumeration vs brute-force subset checking."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bron_kerbosch import (maximal_cliques, maximal_cliques_degeneracy, maximum_clique,
                           brute_maximal_cliques, _adj_sets, _is_clique)

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


# --- known cases ------------------------------------------------------------
# two triangles sharing an edge, plus an isolated vertex
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
bk = set(maximal_cliques(5, edges))
check("two triangles sharing an edge: cliques {012},{123},{4}",
      bk == {frozenset({0, 1, 2}), frozenset({1, 2, 3}), frozenset({4})})
check("maximum clique size is 3", maximum_clique(5, edges)[1] == 3)

# a complete graph K4: single maximal clique of all 4
edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
check("K4 has one maximal clique of size 4", maximal_cliques(4, edges) == [frozenset({0, 1, 2, 3})])

# a path 0-1-2-3: maximal cliques are its edges
edges = [(0, 1), (1, 2), (2, 3)]
check("path graph: maximal cliques are the edges",
      set(maximal_cliques(4, edges)) == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})})

# empty graph: each vertex is its own maximal clique
check("edgeless graph: every vertex is a maximal clique",
      set(maximal_cliques(3, [])) == {frozenset({0}), frozenset({1}), frozenset({2})})

# --- exhaustive validation vs brute ----------------------------------------
rng = LCG(2026)
match_ok = degen_ok = maxc_ok = True
saw_big = False
for _ in range(400):
    n = rng.randint(1, 9)
    edges = random_graph(rng, n, rng.randint(1, 3), 4)   # ~25-75% density
    bk = set(maximal_cliques(n, edges))
    brute = set(brute_maximal_cliques(n, edges))
    if bk != brute:
        match_ok = False
        print(f"  mismatch: bk={sorted(map(sorted, bk))} brute={sorted(map(sorted, brute))} "
              f"n={n} edges={edges}")
        break
    if set(maximal_cliques_degeneracy(n, edges)) != bk:
        degen_ok = False
        break
    # maximum clique size must equal the largest brute maximal clique
    mc = maximum_clique(n, edges)[1]
    if mc != max((len(c) for c in brute), default=0):
        maxc_ok = False
        break
    if mc >= 3:
        saw_big = True
check("Bron-Kerbosch (pivoting) matches brute force (400 random graphs)", match_ok)
check("degeneracy-ordering variant gives identical cliques", degen_ok)
check("maximum-clique size matches the largest brute maximal clique", maxc_ok)
check("suite included graphs with a clique of size >= 3", saw_big)

# --- every returned set is a maximal clique --------------------------------
rng = LCG(4242)
valid_ok = True
for _ in range(200):
    n = rng.randint(1, 9)
    edges = random_graph(rng, n, rng.randint(1, 3), 4)
    adj = _adj_sets(n, edges)
    for c in maximal_cliques(n, edges):
        if not _is_clique(c, adj):
            valid_ok = False
            break
        # no outside vertex is adjacent to all of c
        for u in range(n):
            if u not in c and all(u in adj[w] for w in c):
                valid_ok = False
                break
        if not valid_ok:
            break
    if not valid_ok:
        break
check("every returned set is a clique and truly maximal", valid_ok)

# --- cliques are distinct ---------------------------------------------------
rng = LCG(777)
distinct_ok = True
for _ in range(200):
    n = rng.randint(1, 9)
    edges = random_graph(rng, n, 2, 4)
    cliques = maximal_cliques(n, edges)
    if len(cliques) != len(set(cliques)):
        distinct_ok = False
        break
check("returned maximal cliques are all distinct", distinct_ok)

# --- Moon-Moser: the complete tripartite-style graph has many cliques ------
# the Moon-Moser graph on 3k vertices (k groups of 3, complete between groups) has 3^k maximal
# cliques; check k=3 (9 vertices -> 27 cliques)
k = 3
groups = [[3 * g, 3 * g + 1, 3 * g + 2] for g in range(k)]
mm_edges = []
verts = [v for grp in groups for v in grp]
for i in range(len(verts)):
    for j in range(i + 1, len(verts)):
        u, w = verts[i], verts[j]
        # connect iff in different groups
        if u // 3 != w // 3:
            mm_edges.append((u, w))
check("Moon-Moser graph (k=3) has 3^3 = 27 maximal cliques",
      len(maximal_cliques(9, mm_edges)) == 27)

# --- a bigger sparse graph solves and every clique is valid ----------------
rng = LCG(31337)
n = 60
edges = random_graph(rng, n, 1, 12)          # sparse (~8% density)
cliques = maximal_cliques_degeneracy(n, edges)
adj = _adj_sets(n, edges)
big_ok = all(_is_clique(c, adj) for c in cliques)
check("60-vertex sparse graph: all maximal cliques valid", big_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bron_kerbosch tests passed")

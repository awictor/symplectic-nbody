"""Tests for bipartite_matching: Hopcroft-Karp vs brute force, plus Konig's and Hall's theorems."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bipartite_matching import (BipartiteGraph, hopcroft_karp, minimum_vertex_cover,
                                maximum_independent_set, has_perfect_left_matching,
                                brute_max_matching, hall_condition_holds)

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


def random_bipartite(rng, nl, nr, p_num, p_den):
    g = BipartiteGraph(nl, nr)
    for u in range(nl):
        for v in range(nr):
            if rng.rand() % p_den < p_num:
                g.add_edge(u, v)
    return g


def is_valid_matching(g, match_l, match_r):
    """match_l/match_r are consistent, only use real edges, and are vertex-disjoint."""
    for u in range(g.nl):
        v = match_l[u]
        if v == -1:
            continue
        if v not in g.adj[u]:
            return False
        if match_r[v] != u:
            return False
    for v in range(g.nr):
        u = match_r[v]
        if u != -1 and match_l[u] != v:
            return False
    return True


# --- known small cases ------------------------------------------------------
g = BipartiteGraph(3, 3)
for u, v in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)]:
    g.add_edge(u, v)
size, ml, mr = hopcroft_karp(g)
check("3x3 with a perfect matching: size 3", size == 3)
check("returned matching is valid", is_valid_matching(g, ml, mr))

# a bottleneck: two left vertices both connect only to one right vertex
g = BipartiteGraph(2, 2)
g.add_edge(0, 0)
g.add_edge(1, 0)
size, _, _ = hopcroft_karp(g)
check("shared-only-neighbour bottleneck: size 1", size == 1)
check("bottleneck has no perfect left matching", not has_perfect_left_matching(g))
check("bottleneck violates Hall's condition", not hall_condition_holds(g))

# empty graph: matching 0
g = BipartiteGraph(4, 4)
check("no edges -> matching 0", hopcroft_karp(g)[0] == 0)

# complete bipartite K_{3,4}: matching = min(3,4) = 3
g = BipartiteGraph(3, 4)
for u in range(3):
    for v in range(4):
        g.add_edge(u, v)
check("K(3,4): matching is min(3,4)=3", hopcroft_karp(g)[0] == 3)

# --- exhaustive validation vs brute force ----------------------------------
rng = LCG(20260911)
match_ok = valid_ok = cover_ok = mis_ok = True
saw_imperfect = False
for _ in range(500):
    nl = rng.randint(1, 6)
    nr = rng.randint(1, 6)
    g = random_bipartite(rng, nl, nr, rng.randint(1, 3), 4)   # ~25-75% edge density
    size, ml, mr = hopcroft_karp(g)
    if size != brute_max_matching(g):
        match_ok = False
        break
    if not is_valid_matching(g, ml, mr):
        valid_ok = False
        break
    # Konig: min vertex cover size == max matching size, and it IS a cover
    lc, rc = minimum_vertex_cover(g)
    if len(lc) + len(rc) != size:
        cover_ok = False
        break
    lcs, rcs = set(lc), set(rc)
    for u in range(g.nl):                      # every edge must be covered
        for v in g.adj[u]:
            if u not in lcs and v not in rcs:
                cover_ok = False
                break
        if not cover_ok:
            break
    if not cover_ok:
        break
    # max independent set size == nl + nr - matching
    ls, rs = maximum_independent_set(g)
    if len(ls) + len(rs) != g.nl + g.nr - size:
        mis_ok = False
        break
    if size < min(g.nl, g.nr):
        saw_imperfect = True
check("Hopcroft-Karp size matches brute force on 500 random graphs", match_ok)
check("returned matchings are always valid", valid_ok)
check("Konig: min vertex cover size == max matching, and covers every edge", cover_ok)
check("max independent set size == nl + nr - matching", mis_ok)
check("random suite included imperfect matchings", saw_imperfect)

# --- Hall's theorem: perfect-left-matching iff Hall's condition ------------
rng = LCG(555)
hall_ok = True
for _ in range(300):
    nl = rng.randint(1, 6)
    nr = rng.randint(nl, 7)                    # nr >= nl so a left-perfect matching is possible
    g = random_bipartite(rng, nl, nr, rng.randint(1, 3), 4)
    if has_perfect_left_matching(g) != hall_condition_holds(g):
        hall_ok = False
        break
check("Hall's theorem: left-perfect matching exists iff Hall's condition holds", hall_ok)

# --- augmenting-path optimality: adding an edge never decreases matching ---
rng = LCG(99)
monotone_ok = True
g = BipartiteGraph(6, 6)
prev = 0
for _ in range(30):
    u = rng.randint(0, 5)
    v = rng.randint(0, 5)
    g.add_edge(u, v)
    cur = hopcroft_karp(g)[0]
    if cur < prev:
        monotone_ok = False
        break
    prev = cur
check("matching size is monotone as edges are added", monotone_ok)

# --- larger performance sanity: a big sparse graph solves quickly ----------
rng = LCG(31337)
g = BipartiteGraph(2000, 2000)
for u in range(2000):
    for _ in range(3):                         # ~3 edges each -> sparse
        g.add_edge(u, rng.randint(0, 1999))
size = hopcroft_karp(g)[0]
check("2000x2000 sparse graph solves and matches a good fraction", 1200 < size <= 2000)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bipartite_matching tests passed")

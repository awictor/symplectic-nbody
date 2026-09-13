"""Tests for blossom: size matches brute force, odd cycles handled, valid matchings, perfect matchings."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blossom import (Graph, maximum_matching, matching_size, matched_pairs,  # noqa: E402
                     is_valid_matching, has_augmenting_path, brute_maximum_matching_size)


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


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def nxt(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s

    def randint(self, lo, hi):
        return lo + (self.nxt() >> 8) % (hi - lo + 1)


def cycle(n):
    g = Graph(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def complete(n):
    g = Graph(n)
    for u in range(n):
        for v in range(u + 1, n):
            g.add_edge(u, v)
    return g


def main():
    # ---- 1. the defining odd-cycle cases ---------------------------------------------
    for k, expect in [(3, 1), (5, 2), (7, 3), (9, 4)]:
        g = cycle(k)
        m = maximum_matching(g)
        check(f"{k}-cycle matches {expect} (odd cycle)", matching_size(m) == expect and
              is_valid_matching(g, m), f"got {matching_size(m)}")

    # even cycles have a perfect matching
    for k in (4, 6, 8):
        g = cycle(k)
        m = maximum_matching(g)
        check(f"{k}-cycle has a perfect matching", matching_size(m) == k // 2)

    # ---- 2. the Petersen graph (famous 3-regular, has a perfect matching) -------------
    pet = Graph(10)
    outer = [(i, (i + 1) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    for u, v in outer + spokes + inner:
        pet.add_edge(u, v)
    m = maximum_matching(pet)
    check("Petersen graph has a perfect matching (5 edges)", matching_size(m) == 5 and
          is_valid_matching(pet, m), f"got {matching_size(m)}")

    # ---- 3. exact agreement with brute force over random general graphs ---------------
    rng = LCG(2024)
    mism = 0
    invalid = 0
    for _ in range(300):
        n = rng.randint(2, 9)
        g = Graph(n)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.randint(0, 1):
                    g.add_edge(u, v)
        m = maximum_matching(g)
        if not is_valid_matching(g, m):
            invalid += 1
        if matching_size(m) != brute_maximum_matching_size(g):
            mism += 1
    check("blossom size == brute force over 300 random graphs", mism == 0, f"{mism} mismatches")
    check("all returned matchings are valid", invalid == 0, f"{invalid} invalid")

    # ---- 4. complete graphs: K_n has floor(n/2) matching ------------------------------
    for n in (2, 3, 4, 5, 6, 7):
        g = complete(n)
        m = maximum_matching(g)
        check(f"K_{n} matching size floor(n/2)", matching_size(m) == n // 2, f"{matching_size(m)}")

    # ---- 5. odd complete graph has no perfect matching --------------------------------
    g = complete(5)
    m = maximum_matching(g)
    check("K_5 (odd) has no perfect matching (one vertex left)", matching_size(m) == 2 and
          sum(1 for v in range(5) if m[v] == -1) == 1)

    # ---- 6. bipartite graphs handled correctly ---------------------------------------
    # a path graph (bipartite) on 6 vertices
    g = Graph(6)
    for i in range(5):
        g.add_edge(i, i + 1)
    m = maximum_matching(g)
    check("path graph P6 matches 3 edges", matching_size(m) == 3)
    # a bipartite graph with a known matching
    g = Graph(6)   # left 0,1,2  right 3,4,5
    for u, v in [(0, 3), (0, 4), (1, 4), (2, 5)]:
        g.add_edge(u, v)
    m = maximum_matching(g)
    check("bipartite matching size matches brute", matching_size(m) == brute_maximum_matching_size(g))

    # ---- 7. Berge: no augmenting path remains in the maximum matching -----------------
    rng = LCG(99)
    berge_ok = True
    for _ in range(30):
        n = rng.randint(2, 8)
        g = Graph(n)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.randint(0, 1):
                    g.add_edge(u, v)
        if has_augmenting_path(g, maximum_matching(g)):
            berge_ok = False
    check("no augmenting path remains (Berge optimality)", berge_ok)

    # ---- 8. edge cases ----------------------------------------------------------------
    check("empty graph -> empty matching", matching_size(maximum_matching(Graph(0))) == 0)
    check("no edges -> no matching", matching_size(maximum_matching(Graph(5))) == 0)
    g = Graph(2)
    g.add_edge(0, 1)
    check("single edge matches", matching_size(maximum_matching(g)) == 1)
    # self-loops and duplicate edges ignored
    g = Graph(3)
    g.add_edge(0, 0)
    g.add_edge(0, 1)
    g.add_edge(0, 1)
    check("self-loops and duplicates ignored", matching_size(maximum_matching(g)) == 1)

    # ---- 9. matched_pairs consistency -------------------------------------------------
    g = cycle(6)
    m = maximum_matching(g)
    pairs = matched_pairs(m)
    check("matched_pairs count equals matching size", len(pairs) == matching_size(m))
    check("matched_pairs are real edges", all((u, v) in g._edges for u, v in pairs))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for chinese_postman: route = edges + min matching, Eulerian circuits, brute-force matching."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chinese_postman import (chinese_postman, eulerian_circuit, is_eulerian_circuit,  # noqa: E402
                             brute_min_matching, _all_pairs_shortest, _build_adj, INF)


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


def main():
    # ---- 1. Eulerian square: no retracing ---------------------------------------------
    sq = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]
    r = chinese_postman(4, sq)
    check("square is Eulerian", r["is_eulerian"] and r["extra_cost"] == 0)
    check("square route = total edges", r["route_length"] == 4.0)
    c = eulerian_circuit(4, sq)
    check("square Eulerian circuit valid", is_eulerian_circuit(sq, c))

    # ---- 2. path graph: forces retracing ----------------------------------------------
    path = [(0, 1, 1), (1, 2, 1)]
    r = chinese_postman(3, path)
    check("path has odd endpoints", set(r["odd_vertices"]) == {0, 2})
    check("path route = edges + retrace whole path", r["route_length"] == 4.0 and r["extra_cost"] == 2.0)
    check("path is not Eulerian", not r["is_eulerian"])

    # ---- 3. route length = total edge weight + min matching cost ----------------------
    rng = LCG(2024)
    formula_bad = 0
    matching_bad = 0
    for _ in range(200):
        n = rng.randint(3, 7)
        # random connected graph: spanning path + extra edges
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1, rng.randint(1, 9)))
        for _ in range(rng.randint(0, n)):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u != v:
                edges.append((u, v, rng.randint(1, 9)))
        r = chinese_postman(n, edges)
        adj, degree, total, present = _build_adj(n, edges)
        odds = [i for i in range(n) if degree[i] % 2]
        dist = _all_pairs_shortest(n, adj)
        brute = brute_min_matching(odds, dist)
        if abs(r["extra_cost"] - brute) > 1e-9:
            matching_bad += 1
        if abs(r["route_length"] - (total + brute)) > 1e-9:
            formula_bad += 1
    check("extra cost == brute-force min matching (200 graphs)", matching_bad == 0, f"{matching_bad}")
    check("route length == total edges + min matching", formula_bad == 0, f"{formula_bad}")

    # ---- 4. handshake lemma: even number of odd-degree vertices -----------------------
    handshake_ok = True
    for _ in range(100):
        n = rng.randint(2, 8)
        edges = [(rng.randint(0, n - 1), rng.randint(0, n - 1), 1) for _ in range(rng.randint(1, 12))]
        edges = [(u, v, w) for u, v, w in edges if u != v]
        _, degree, _, _ = _build_adj(n, edges)
        odd_count = sum(1 for d in degree if d % 2)
        if odd_count % 2 != 0:
            handshake_ok = False
    check("always an even number of odd-degree vertices (handshake lemma)", handshake_ok)

    # ---- 5. Eulerian circuit uses every edge exactly once (random even graphs) --------
    circuit_bad = 0
    trials = 0
    for _ in range(100):
        n = rng.randint(3, 7)
        # build an even-degree connected graph: a cycle plus doubled random edges keeps parity even
        edges = [(i, (i + 1) % n, rng.randint(1, 5)) for i in range(n)]   # cycle -> all degree 2
        # add pairs of parallel edges to keep even
        for _ in range(rng.randint(0, 3)):
            u = rng.randint(0, n - 1)
            v = (u + rng.randint(1, n - 1)) % n
            w = rng.randint(1, 5)
            edges.append((u, v, w))
            edges.append((u, v, w))          # duplicate keeps both degrees even
        c = eulerian_circuit(n, edges)
        if c is not None:
            trials += 1
            if not is_eulerian_circuit(edges, c):
                circuit_bad += 1
    check(f"Eulerian circuits valid on {trials} even graphs", circuit_bad == 0, f"{circuit_bad}")

    # ---- 6. odd-degree graph has no Eulerian circuit ----------------------------------
    check("path graph has no Eulerian circuit", eulerian_circuit(3, path) is None)

    # ---- 7. disconnected graph -> no postman route ------------------------------------
    disc = [(0, 1, 1), (2, 3, 1)]
    r = chinese_postman(4, disc)
    check("disconnected graph -> infinite route", r["route_length"] == INF)

    # ---- 8. a known asymmetric instance -----------------------------------------------
    # a "bowtie": two triangles sharing a vertex; all vertices even except... check
    edges = [(0, 1, 2), (1, 2, 2), (2, 0, 2), (2, 3, 3), (3, 4, 3), (4, 2, 3)]
    r = chinese_postman(5, edges)
    check("bowtie (all even) is Eulerian", r["is_eulerian"] and r["extra_cost"] == 0
          and r["route_length"] == 15.0)

    # single retrace instance: a triangle with one extra spur (vertices 3 and its neighbour odd)
    edges = [(0, 1, 1), (1, 2, 1), (2, 0, 1), (0, 3, 5)]   # 0 has degree 3, 3 has degree 1 -> odd {0,3}
    r = chinese_postman(4, edges)
    check("spur must be retraced", set(r["odd_vertices"]) == {0, 3} and r["extra_cost"] == 5.0
          and r["route_length"] == 13.0)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

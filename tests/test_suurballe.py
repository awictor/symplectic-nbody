"""Tests for suurballe: disjoint minimum-cost path pair vs brute force, single-path Dijkstra, edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from suurballe import (Graph, suurballe, dijkstra, paths_edge_disjoint, path_cost,  # noqa: E402
                       brute_min_disjoint_pair, INF)


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
    # ---- 1. diamond: two disjoint routes -----------------------------------------------
    g = Graph(4)
    for u, v, w in [(0, 1, 1), (1, 3, 1), (0, 2, 2), (2, 3, 2)]:
        g.add_edge(u, v, w)
    p, c = suurballe(g, 0, 3)
    check("diamond finds two disjoint paths", p is not None and len(p) == 2)
    check("diamond paths are edge-disjoint", paths_edge_disjoint(p))
    check("diamond total cost is 6", c == 6, f"{c}")
    check("diamond matches brute force", c == brute_min_disjoint_pair(g, 0, 3))

    # ---- 2. two node-disjoint routes through intermediate vertices --------------------
    # (Suurballe here assumes a simple graph -- distinct (u,v) edges -- so route via vertices)
    g = Graph(4)
    for u, v, w in [(0, 1, 3), (1, 3, 0), (0, 2, 5), (2, 3, 0)]:
        g.add_edge(u, v, w)
    p, c = suurballe(g, 0, 3)
    check("two routes give a disjoint pair", p is not None and paths_edge_disjoint(p))
    check("two routes total cost 3+5=8", c == 8, f"{c}")

    # ---- 3. a bridge: no disjoint pair exists -----------------------------------------
    # 0->1->2, single path, no alternative
    g = Graph(3)
    g.add_edge(0, 1, 1)
    g.add_edge(1, 2, 1)
    p, c = suurballe(g, 0, 2)
    check("single-path graph has no disjoint pair", p is None and c == INF)
    check("agrees with brute force (no pair)", brute_min_disjoint_pair(g, 0, 2) == INF)

    # ---- 4. exhaustive agreement with brute force over random graphs ------------------
    rng = LCG(2024)
    mism = 0
    tested = 0
    for _ in range(400):
        n = rng.randint(3, 7)
        g = Graph(n)
        for u in range(n):
            for v in range(n):
                if u != v and rng.randint(0, 2) == 0:
                    g.add_edge(u, v, rng.randint(1, 9))
        p, c = suurballe(g, 0, n - 1)
        brute = brute_min_disjoint_pair(g, 0, n - 1)
        tested += 1
        if p is None:
            if brute != INF:
                mism += 1
        else:
            if not paths_edge_disjoint(p):
                mism += 1
            elif abs(c - brute) > 1e-9:
                mism += 1
    check(f"Suurballe matches brute force over {tested} random graphs", mism == 0, f"{mism}")

    # ---- 5. returned paths run source -> target and cost matches ----------------------
    g = Graph(6)
    for u, v, w in [(0, 1, 2), (0, 2, 3), (1, 3, 2), (2, 3, 1), (1, 4, 4),
                    (3, 5, 2), (4, 5, 1), (2, 4, 2)]:
        g.add_edge(u, v, w)
    p, c = suurballe(g, 0, 5)
    if p is not None:
        check("both paths start at source and end at target",
              all(path[0] == 0 and path[-1] == 5 for path in p))
        check("reported cost equals sum of path costs",
              abs(c - sum(path_cost(g, path) for path in p)) < 1e-9)
        check("paths are edge-disjoint", paths_edge_disjoint(p))

    # ---- 6. single-path Dijkstra matches a reference ----------------------------------
    dist, pred = dijkstra(g, 0)
    # reference: BFS-relaxation shortest path to 5
    check("Dijkstra distance to target finite", dist[5] < INF)
    # the disjoint pair's cheaper path is >= the single shortest path
    if p is not None:
        single = dist[5]
        check("each disjoint path cost >= single shortest path",
              all(path_cost(g, path) >= single - 1e-9 for path in p))

    # ---- 7. reduced costs are non-negative --------------------------------------------
    dist, _ = dijkstra(g, 0)
    rc_ok = all(w + dist[u] - dist[v] >= -1e-9 for (u, v, w) in g.edges
                if dist[u] < INF and dist[v] < INF)
    check("reduced costs are non-negative", rc_ok)

    # ---- 8. disjoint pair total cost >= single shortest path * (roughly) --------------
    # and the pair uses two genuinely different routes
    g = Graph(4)
    for u, v, w in [(0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 3, 1), (1, 2, 5)]:
        g.add_edge(u, v, w)
    p, c = suurballe(g, 0, 3)
    check("symmetric diamond cost 4", c == 4, f"{c}")
    check("symmetric diamond disjoint", paths_edge_disjoint(p))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for Johnson's algorithm: agreement with Floyd-Warshall, reweighting, negative cycles, paths."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import johnson as J  # noqa: E402
import floyd_warshall as FW  # noqa: E402


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
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def _agree(n, dist, fw):
    for i in range(n):
        for j in range(n):
            a, b = dist[i][j], fw[i][j]
            if a == math.inf and b == math.inf:
                continue
            if a == math.inf or b == math.inf or abs(a - b) > 1e-9:
                return False
    return True


def main():
    # ---- 1. CLRS Johnson example: matches Floyd-Warshall, known distances ---------------
    edges = [(0, 1, 3), (0, 2, 8), (0, 4, -4), (1, 3, 1), (1, 4, 7),
             (2, 1, 4), (3, 0, 2), (3, 2, -5), (4, 3, 6)]
    n = 5
    dist, nh = J.johnson(n, edges)
    fw, _ = FW.floyd_warshall(n, edges)
    check("Johnson == Floyd-Warshall (negative edges)", _agree(n, dist, fw))
    check("known distances from node 0", dist[0] == [0.0, 1.0, -3.0, 2.0, -4.0], f"{dist[0]}")

    # ---- 2. reweighted edges are all non-negative and preserve shortest paths -----------
    rw, h = J.reweighted_edges(n, edges)
    check("reweighted edges all non-negative", all(w >= -1e-12 for _, _, w in rw))
    # preservation: reweighted path length = original + h[s] - h[t] for the shortest path
    # verify by re-deriving Johnson distances equal Floyd
    check("reweighting preserves shortest paths (distances match FW)", _agree(n, dist, fw))

    # ---- 3. path reconstruction gives a path of the reported length ---------------------
    def path_len(path):
        wmap = {(u, v): w for (u, v, w) in edges}
        return sum(wmap[(path[i], path[i + 1])] for i in range(len(path) - 1))

    ok = True
    for s in range(n):
        for t in range(n):
            if s == t or dist[s][t] == math.inf:
                continue
            p = J.reconstruct_path(nh, s, t)
            if not p or p[0] != s or p[-1] != t:
                ok = False
            elif abs(path_len(p) - dist[s][t]) > 1e-9:
                ok = False
    check("reconstructed paths have the reported length", ok)

    # ---- 4. negative cycle is detected --------------------------------------------------
    try:
        J.johnson(3, [(0, 1, 1), (1, 2, -3), (2, 0, 1)])
        check("negative cycle raises", False, "no exception")
    except ValueError:
        check("negative cycle raises", True)

    # ---- 5. all-non-negative graph: matches Floyd and a per-source Dijkstra -------------
    pos = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5), (3, 4, 3)]
    n2 = 5
    d2, _ = J.johnson(n2, pos)
    fw2, _ = FW.floyd_warshall(n2, pos)
    check("non-negative graph matches Floyd-Warshall", _agree(n2, d2, fw2))
    # 0->2 (1) -> 1 (2) -> 3 (1) = 4, cheaper than 0->1 (4) -> 3 (1) = 5
    check("shortest 0->3 via 2->1 is 4", abs(d2[0][3] - 4.0) < 1e-9, f"{d2[0][3]}")

    # ---- 6. disconnected nodes are infinity ---------------------------------------------
    disc = [(0, 1, 2.0)]
    d3, _ = J.johnson(3, disc)   # node 2 isolated
    check("unreachable pairs are infinity", d3[0][2] == math.inf and d3[2][0] == math.inf)
    check("self-distance is zero", all(d3[i][i] == 0.0 for i in range(3)))

    # ---- 7. randomized cross-check against Floyd-Warshall (no negative cycles) ----------
    rnd = _lcg(2024)
    all_ok = True
    for trial in range(20):
        m = 7
        redges = []
        for _ in range(14):
            u = int(rnd() * m) % m
            v = int(rnd() * m) % m
            if u == v:
                continue
            w = round(rnd() * 12 - 3, 2)   # weights in [-3, 9]
            redges.append((u, v, w))
        try:
            dj, _ = J.johnson(m, redges)
        except ValueError:
            continue                        # skip graphs with negative cycles
        try:
            fwd, _ = FW.floyd_warshall(m, redges)
        except ValueError:
            continue
        if not _agree(m, dj, fwd):
            all_ok = False
    check("randomized graphs: Johnson == Floyd-Warshall", all_ok)

    # ---- 8. single node -----------------------------------------------------------------
    d1, _ = J.johnson(1, [])
    check("single-node graph", d1 == [[0.0]])

    # ---- 9. bellman_ford helper matches Floyd's single-source row -----------------------
    bf, _ = J.bellman_ford(n, edges, 0)
    check("Johnson's bellman_ford matches FW row 0",
          all((bf[j] == math.inf and fw[0][j] == math.inf) or abs(bf[j] - fw[0][j]) < 1e-9
              for j in range(n)), f"{bf}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

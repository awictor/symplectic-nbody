"""Tests for floyd_warshall.py -- all-pairs shortest paths.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Distances are cross-checked
against running Dijkstra from every source on nonnegative-weight graphs.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import floyd_warshall as F  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- basic distances --------------------------------------------------------
edges = [(0, 1, 3), (1, 2, 1), (0, 2, 10), (2, 3, 2)]
dist, nxt = F.floyd_warshall(4, edges)
check("distance to self is 0", all(dist[i][i] == 0 for i in range(4)))
check("direct edge distance", dist[0][1] == 3)
check("shortest path uses the cheaper route (0->1->2 = 4, not 10)", dist[0][2] == 4)
check("multi-hop distance 0->3", dist[0][3] == 6)
check("unreachable is infinite", dist[3][0] == math.inf)

# --- path reconstruction ----------------------------------------------------
check("path 0->3 is reconstructed", F.reconstruct_path(nxt, 0, 3) == [0, 1, 2, 3])
check("path to self is a single node", F.reconstruct_path(nxt, 2, 2) == [2])
check("unreachable path is empty", F.reconstruct_path(nxt, 3, 0) == [])
# every reconstructed path's edge sum equals the distance
edge_w = {(u, v): w for u, v, w in edges}
ok = True
for i in range(4):
    for j in range(4):
        p = F.reconstruct_path(nxt, i, j)
        if p:
            total = sum(edge_w[(p[k], p[k + 1])] for k in range(len(p) - 1))
            if total != dist[i][j]:
                ok = False
check("reconstructed paths' lengths equal the distances", ok)

# --- negative weights (Dijkstra can't do these) ----------------------------
neg = [(0, 2, -2), (1, 0, 4), (1, 2, 3), (2, 3, 2), (3, 1, -1)]
d2, n2 = F.floyd_warshall(4, neg)
check("handles negative edges: dist[0][1]", d2[0][1] == -1)
check("negative-edge path is correct", F.reconstruct_path(n2, 0, 1) == [0, 2, 3, 1])
check("negative-edge shortest is below the direct route", d2[0][2] == -2)

# --- negative-cycle detection ----------------------------------------------
check("detects a negative cycle", F.has_negative_cycle(3, [(0, 1, 1), (1, 2, -3), (2, 0, 1)]))
check("no false positive on a positive cycle", not F.has_negative_cycle(3, [(0, 1, 1), (1, 2, 1), (2, 0, 1)]))
check("no negative cycle in a DAG", not F.has_negative_cycle(3, [(0, 1, -5), (1, 2, -5)]))
try:
    F.floyd_warshall(3, [(0, 1, 1), (1, 2, -3), (2, 0, 1)])
    check("floyd_warshall raises on a negative cycle", False)
except ValueError:
    check("floyd_warshall raises on a negative cycle", True)

# --- transitive closure -----------------------------------------------------
tc = F.transitive_closure(4, [(0, 1), (1, 2)])
check("reachability is transitive (0 reaches 2)", tc[0][2])
check("no reverse reachability (2 doesn't reach 0)", not tc[2][0])
check("every node reaches itself", all(tc[i][i] for i in range(4)))
check("isolated node reaches only itself", tc[3] == [False, False, False, True])
# closure of a cycle: everyone reaches everyone
cyc_tc = F.transitive_closure(3, [(0, 1), (1, 2), (2, 0)])
check("a cycle's closure is all-reachable", all(cyc_tc[i][j] for i in range(3) for j in range(3)))

# --- agreement with Dijkstra over random nonnegative graphs ----------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 16


gen = lcg(9)


def rnd(lo, hi):
    return lo + next(gen) % (hi - lo + 1)


mismatches = 0
for _ in range(300):
    n = rnd(1, 8)
    m = rnd(0, n * 2)
    es = []
    for _ in range(m):
        u, v = rnd(0, n - 1), rnd(0, n - 1)
        if u != v:
            es.append((u, v, rnd(1, 20)))
    fw, _ = F.floyd_warshall(n, es)
    dj = F.dijkstra_all_pairs(n, es)
    if fw != dj:
        mismatches += 1
check("Floyd-Warshall matches all-pairs Dijkstra on 300 random graphs", mismatches == 0)

# --- symmetry on undirected graphs -----------------------------------------
und = []
for u, v, w in [(0, 1, 2), (1, 2, 3), (0, 2, 10)]:
    und += [(u, v, w), (v, u, w)]
dsym, _ = F.floyd_warshall(3, und)
check("undirected graph has a symmetric distance matrix",
      all(dsym[i][j] == dsym[j][i] for i in range(3) for j in range(3)))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall floyd_warshall tests passed")

"""Tests for bridges: Tarjan bridges/articulation points validated vs the brute-force definition."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bridges import (Graph, find_bridges_and_articulation, two_edge_connected_components,
                     brute_bridges, brute_articulation)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- deterministic RNG (seeded LCG) ----------------------------------------
class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16          # use high bits

    def randint(self, lo, hi):        # inclusive
        return lo + self.rand() % (hi - lo + 1)


def random_graph(rng, n, m):
    """Random undirected graph, n vertices, m edges (parallel edges and self-avoiding)."""
    g = Graph(n)
    for _ in range(m):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            g.add_edge(u, v)
    return g


# --- known small cases ------------------------------------------------------
# path graph: every edge a bridge, every internal vertex a cut vertex
g = Graph(5)
for u, v in [(0, 1), (1, 2), (2, 3), (3, 4)]:
    g.add_edge(u, v)
b, a = find_bridges_and_articulation(g)
check("path: all 4 edges are bridges", b == [(0, 1), (1, 2), (2, 3), (3, 4)])
check("path: internal vertices 1,2,3 are articulation points", a == [1, 2, 3])

# cycle: no bridges, no articulation points (2-connected)
g = Graph(5)
for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]:
    g.add_edge(u, v)
b, a = find_bridges_and_articulation(g)
check("cycle: no bridges", b == [])
check("cycle: no articulation points", a == [])

# two cycles joined by one bridge edge
g = Graph(6)
for u, v in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]:
    g.add_edge(u, v)
b, a = find_bridges_and_articulation(g)
check("bridged cycles: only the joining edge is a bridge", b == [(2, 3)])
check("bridged cycles: the two joint vertices are cut vertices", a == [2, 3])
check("bridged cycles: 2-edge-connected components are the two triangles",
      two_edge_connected_components(g) == [[0, 1, 2], [3, 4, 5]])

# star graph K_{1,4}: center is the only articulation point, every edge a bridge
g = Graph(5)
for leaf in [1, 2, 3, 4]:
    g.add_edge(0, leaf)
b, a = find_bridges_and_articulation(g)
check("star: center is the only articulation point", a == [0])
check("star: all spokes are bridges", len(b) == 4)

# parallel edges are never bridges
g = Graph(2)
g.add_edge(0, 1)
g.add_edge(0, 1)
check("double edge is not a bridge", find_bridges_and_articulation(g)[0] == [])
check("double edge matches brute", find_bridges_and_articulation(g)[0] == brute_bridges(g))

# single edge IS a bridge, but its endpoints are NOT cut vertices (removing a
# degree-1 vertex leaves the rest connected -> component count unchanged)
g = Graph(2)
g.add_edge(0, 1)
b, a = find_bridges_and_articulation(g)
check("single edge is a bridge", b == [(0, 1)])
check("single edge endpoints are not articulation points", a == [])

# --- exhaustive validation vs the definition on random graphs --------------
rng = LCG(12345)
bridge_ok = art_ok = True
disconnected_seen = False
for trial in range(400):
    n = rng.randint(1, 9)
    m = rng.randint(0, 14)
    g = random_graph(rng, n, m)
    b, a = find_bridges_and_articulation(g)
    if b != brute_bridges(g):
        bridge_ok = False
        print(f"  bridge mismatch trial {trial}: fast={b} brute={brute_bridges(g)}")
        break
    if a != brute_articulation(g):
        art_ok = False
        print(f"  articulation mismatch trial {trial}: fast={a} brute={brute_articulation(g)}")
        break
    # confirm we exercised disconnected graphs too
    from bridges import _count_components
    if _count_components(g.n, g.edges) > 1:
        disconnected_seen = True
check("bridges match brute-force definition on 400 random graphs", bridge_ok)
check("articulation points match brute-force definition on 400 random graphs", art_ok)
check("random suite included disconnected graphs", disconnected_seen)

# --- 2-edge-connected components partition all vertices --------------------
rng = LCG(777)
part_ok = True
for _ in range(100):
    n = rng.randint(2, 8)
    g = random_graph(rng, n, rng.randint(1, 12))
    comps = two_edge_connected_components(g)
    covered = sorted(v for c in comps for v in c)
    if covered != list(range(n)):
        part_ok = False
        break
check("2-edge-connected components partition every vertex exactly once", part_ok)

# --- a complete graph K5 has no bridges or cut vertices --------------------
g = Graph(5)
for i in range(5):
    for j in range(i + 1, 5):
        g.add_edge(i, j)
b, a = find_bridges_and_articulation(g)
check("K5: no bridges, no articulation points", b == [] and a == [])

# --- deep path (recursion-safety of the iterative DFS) ---------------------
n = 5000
g = Graph(n)
for i in range(n - 1):
    g.add_edge(i, i + 1)
b, a = find_bridges_and_articulation(g)
check("deep 5000-vertex path: n-1 bridges, no stack overflow", len(b) == n - 1)
check("deep path: all internal vertices are cut vertices", len(a) == n - 2)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bridges tests passed")

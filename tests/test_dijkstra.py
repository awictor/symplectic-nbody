"""Tests for dijkstra.py -- shortest paths and the from-scratch min-heap.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Distances are cross-checked
against Bellman-Ford on random graphs; the heap is checked as a sorter.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dijkstra as D  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- the min-heap ----------------------------------------------------------
h = D.MinHeap()
check("empty heap has length 0", len(h) == 0)
for v in [5, 3, 8, 1, 9, 2, 7]:
    h.push(v, v)
check("heap length tracks pushes", len(h) == 7)
popped = [h.pop()[0] for _ in range(7)]
check("heap pops in sorted order", popped == sorted([5, 3, 8, 1, 9, 2, 7]))
check("emptied heap has length 0", len(h) == 0)
try:
    h.pop()
    check("pop from empty raises", False)
except IndexError:
    check("pop from empty raises", True)
# heap sorts a deterministic pseudo-random stream
hp = D.MinHeap()
state = 7
vals = []
for _ in range(500):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    x = (state >> 16) % 10000
    vals.append(x)
    hp.push(x, x)
check("heap sorts 500 random values", [hp.pop()[0] for _ in range(500)] == sorted(vals))

# --- classic shortest path -------------------------------------------------
g = D.Graph(undirected=True)
for u, v, w in [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]:
    g.add_edge(u, v, w)
dist, prev = D.dijkstra(g, 0)
check("source distance is 0", dist[0] == 0.0)
check("shortest distances are correct", (dist[1], dist[2], dist[3]) == (3.0, 1.0, 4.0))
path, cost = D.shortest_path(g, 0, 3)
check("shortest path is reconstructed", path == [0, 2, 1, 3])
check("path cost matches the distance", cost == 4.0)
check("path from a node to itself is just that node", D.shortest_path(g, 0, 0) == ([0], 0.0))

# --- unreachable and validation --------------------------------------------
g2 = D.Graph()
g2.add_edge(0, 1, 5)
g2.add_node(9)
check("unreachable node has infinite distance", D.dijkstra(g2, 0)[0][9] == math.inf)
check("unreachable path is empty with inf cost", D.shortest_path(g2, 0, 9) == ([], math.inf))
try:
    g2.add_edge(0, 2, -3)
    check("rejects negative edge weight", False)
except ValueError:
    check("rejects negative edge weight", True)
try:
    D.dijkstra(g2, 999)
    check("rejects unknown source", False)
except KeyError:
    check("rejects unknown source", True)

# --- directed graph --------------------------------------------------------
dg = D.Graph(undirected=False)
dg.add_edge("a", "b", 1)
dg.add_edge("b", "c", 1)
dg.add_edge("a", "c", 5)
dist, _ = D.dijkstra(dg, "a")
check("directed shortest path uses the cheaper route", dist["c"] == 2.0)
check("directed edges are one-way (no b->a)", D.dijkstra(dg, "b")[0].get("a", math.inf) == math.inf)

# --- Bellman-Ford agreement on random graphs -------------------------------
def random_graph(seed, n, m, undirected):
    g = D.Graph(undirected=undirected)
    for i in range(n):
        g.add_node(i)
    state = seed
    for _ in range(m):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        u = (state >> 16) % n
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        v = (state >> 16) % n
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        w = 1 + (state >> 16) % 20
        if u != v:
            g.add_edge(u, v, w)
    return g


agree = True
for seed in (1, 2, 3, 4, 5):
    g = random_graph(seed, 30, 120, undirected=(seed % 2 == 0))
    dd, _ = D.dijkstra(g, 0)
    bf = D.bellman_ford(g, 0)
    if any(dd[k] != bf[k] for k in dd):
        agree = False
        break
check("Dijkstra matches Bellman-Ford across random graphs", agree)

# --- triangle inequality holds along shortest paths ------------------------
g = random_graph(11, 25, 100, undirected=False)
dd, _ = D.dijkstra(g, 0)
tri_ok = all(dd[u] + w >= dd[v] - 1e-9 for u, v, w in g.edges() if dd[u] < math.inf)
check("no edge can improve a settled distance (triangle inequality)", tri_ok)

# --- distances are nonnegative and the source is the minimum ---------------
check("all distances are nonnegative", all(d >= 0 for d in dd.values() if d < math.inf))
check("source has the minimum distance (0)", min(d for d in dd.values()) == 0.0)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall dijkstra tests passed")

"""Tests for union_find.py -- disjoint-set union (Union-Find).

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Connectivity is cross-checked
against a brute-force flood fill; Kruskal is checked against known minimum spanning trees.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import union_find as U  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- basics -----------------------------------------------------------------
uf = U.UnionFind(10)
check("n singletons start as n sets", uf.count() == 10)
check("every element is its own root initially", all(uf.find(i) == i for i in range(10)))
check("an element is connected to itself", uf.connected(3, 3))
check("distinct elements start disconnected", not uf.connected(0, 1))
try:
    U.UnionFind(-1)
    check("rejects negative size", False)
except ValueError:
    check("rejects negative size", True)

# --- union / find -----------------------------------------------------------
check("union of separate sets returns True", uf.union(0, 1) is True)
check("union of already-connected returns False", uf.union(0, 1) is False)
uf.union(2, 3)
uf.union(1, 3)  # merges {0,1} with {2,3}
check("transitive connectivity holds", uf.connected(0, 3))
check("still disconnected from an untouched element", not uf.connected(0, 4))
check("merging reduces the set count", uf.count() == 10 - 3)
check("set size is tracked", uf.set_size(0) == 4)
check("components partition all elements",
      sorted(x for g in uf.components() for x in g) == list(range(10)))
check("the big component is {0,1,2,3}", [0, 1, 2, 3] in uf.components())

# --- union is symmetric and idempotent -------------------------------------
uf2 = U.UnionFind(5)
uf2.union(4, 0)
check("union is symmetric (order doesn't matter)", uf2.connected(0, 4))
check("re-union does not change the count", (lambda c: (uf2.union(0, 4), uf2.count() == c)[1])(uf2.count()))

# --- path compression keeps trees flat -------------------------------------
chain = U.UnionFind(100)
for i in range(99):
    chain.union(i, i + 1)  # a long chain
# after a find, everyone on the path points near the root
root = chain.find(0)
chain.find(99)
compressed = sum(1 for i in range(100) if chain.parent[i] == root)
check("all 100 elements are one set", chain.count() == 1)
check("path compression flattens the tree", compressed >= 50)

# --- connected_components helper -------------------------------------------
check("a graph with 3 components counts 3",
      U.connected_components(6, [(0, 1), (1, 2), (3, 4)]) == 3)
check("no edges means n components", U.connected_components(5, []) == 5)
check("a fully connected chain is 1 component",
      U.connected_components(5, [(0, 1), (1, 2), (2, 3), (3, 4)]) == 1)

# --- cross-check against brute-force flood fill ----------------------------
def flood_components(n, edges):
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = [False] * n
    comps = 0
    for s in range(n):
        if not seen[s]:
            comps += 1
            stack = [s]
            while stack:
                x = stack.pop()
                if seen[x]:
                    continue
                seen[x] = True
                stack.extend(y for y in adj[x] if not seen[y])
    return comps


# deterministic pseudo-random graph
n = 60
edges = []
state = 42
for _ in range(50):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    a = (state >> 16) % n
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    b = (state >> 16) % n
    edges.append((a, b))
check("component count matches brute-force flood fill",
      U.connected_components(n, edges) == flood_components(n, edges))

# --- Kruskal minimum spanning tree -----------------------------------------
# a small graph with a known MST
we = [(1, 0, 1), (2, 1, 2), (3, 0, 2), (4, 2, 3), (5, 1, 3)]
total, chosen = U.kruskal_mst(4, we)
check("Kruskal MST weight is minimal (7)", total == 7)
check("MST has n-1 edges", len(chosen) == 3)
check("MST edges form a spanning tree (single component)",
      U.connected_components(4, [(u, v) for _, u, v in chosen]) == 1)
# a disconnected graph yields a spanning forest, not a full tree
forest_total, forest_edges = U.kruskal_mst(5, [(1, 0, 1), (2, 3, 4)])
check("disconnected graph gives a spanning forest", len(forest_edges) == 2)
check("MST never adds a cycle-forming edge",
      all(U.connected_components(4, [(u, v) for _, u, v in chosen[:k]]) == 4 - k
          for k in range(len(chosen) + 1)))
# MST weight is <= any spanning tree: compare to a deliberately worse tree (0-1,1-2,2-3 = 1+2+4=7 here equals; use another)
we2 = [(10, 0, 1), (1, 1, 2), (1, 2, 3), (1, 0, 3), (10, 0, 2)]
mst2, _ = U.kruskal_mst(4, we2)
check("Kruskal picks the cheap cycle-free edges (weight 3)", mst2 == 3)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall union_find tests passed")

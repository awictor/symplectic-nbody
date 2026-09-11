"""Tests for sparse_table: RMQ/max/gcd vs brute force, LCA vs ancestor walk, tree distance vs BFS."""

import os
import sys
from collections import deque
from math import gcd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sparse_table import SparseTable, RMQ, LCA, range_gcd_table

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 55
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- RMQ matches a brute-force scan over every subrange --------------------
a = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]
r = RMQ(a)
ok = True
for l in range(len(a)):
    for rr in range(l + 1, len(a) + 1):
        if r.min_range(l, rr) != min(a[l:rr]):
            ok = False
check("RMQ matches brute force over every subrange", ok)

# --- max sparse table ------------------------------------------------------
mx = SparseTable(a, op=max)
ok = True
for l in range(len(a)):
    for rr in range(l + 1, len(a) + 1):
        if mx.query(l, rr) != max(a[l:rr]):
            ok = False
check("max sparse table matches brute force", ok)

# --- gcd sparse table ------------------------------------------------------
g = range_gcd_table([12, 18, 24, 6, 30, 15])
gd = [12, 18, 24, 6, 30, 15]
ok = True
for l in range(len(gd)):
    for rr in range(l + 1, len(gd) + 1):
        want = 0
        for x in gd[l:rr]:
            want = gcd(want, x)
        if g.query(l, rr) != want:
            ok = False
check("gcd sparse table matches brute force", ok)

# --- randomized RMQ over many arrays ---------------------------------------
ok = True
for _ in range(30):
    arr = [int(rng() * 1000) for _ in range(1 + int(rng() * 60))]
    st = RMQ(arr)
    for _ in range(40):
        l = int(rng() * len(arr))
        rr = l + 1 + int(rng() * (len(arr) - l))
        if st.min_range(l, rr) != min(arr[l:rr]):
            ok = False
            break
    if not ok:
        break
check("randomized RMQ matches brute force over 30 arrays", ok)

# --- single element and full range -----------------------------------------
single = RMQ([42])
check("single-element RMQ", single.min_range(0, 1) == 42)
check("full-range RMQ", r.min_range(0, len(a)) == min(a))
raised = False
try:
    r.query(3, 3)
except IndexError:
    raised = True
check("empty range raises", raised)

# --- LCA against a naive ancestor walk -------------------------------------
def build_random_tree(n):
    edges = []
    parent = [0] * n
    for v in range(1, n):
        p = int(rng() * v)      # attach to an earlier node -> valid tree
        parent[v] = p
        edges.append((p, v))
    return edges, parent


def naive_lca(parent, depth, u, v):
    # walk both up to equal depth, then together
    while depth[u] > depth[v]:
        u = parent[u]
    while depth[v] > depth[u]:
        v = parent[v]
    while u != v:
        u = parent[u]
        v = parent[v]
    return u


def compute_depths(parent, n):
    depth = [0] * n
    for v in range(1, n):
        # depth via walking to root (n small)
        d = 0
        x = v
        while x != 0:
            x = parent[x]
            d += 1
        depth[v] = d
    return depth


def bfs_distance(n, edges, s, t):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    dist = [-1] * n
    dist[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist[t]


lca_ok = dist_ok = True
for _ in range(20):
    n = 5 + int(rng() * 40)
    edges, parent = build_random_tree(n)
    depth = compute_depths(parent, n)
    lca = LCA(n, edges, root=0)
    for _ in range(60):
        u = int(rng() * n)
        v = int(rng() * n)
        if lca.query(u, v) != naive_lca(parent, depth, u, v):
            lca_ok = False
        if lca.distance(u, v) != bfs_distance(n, edges, u, v):
            dist_ok = False
    if not (lca_ok and dist_ok):
        break
check("LCA matches a naive ancestor walk over 20 random trees", lca_ok)
check("tree distance matches BFS shortest path", dist_ok)

# --- LCA on a known small tree ---------------------------------------------
# 0 - 1 - 3, 1 - 4, 0 - 2 - 5
edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
lca = LCA(6, edges)
check("lca(3,4) is their parent 1", lca.query(3, 4) == 1)
check("lca(3,5) is the root 0", lca.query(3, 5) == 0)
check("lca(v,v) is v", lca.query(4, 4) == 4)
check("lca(child,parent) is the parent", lca.query(3, 1) == 1)
check("distance(3,5) is 4", lca.distance(3, 5) == 4)
check("distance to self is 0", lca.distance(4, 4) == 0)
check("kth ancestor: 2nd ancestor of 3 is 0", lca.kth_ancestor(3, 2) == 0)

# --- a path graph (degenerate deep tree) -----------------------------------
path_edges = [(i, i + 1) for i in range(20)]
lca_path = LCA(21, path_edges)
check("path lca(5,15) is 5 (the shallower)", lca_path.query(5, 15) == 5)
check("path distance(5,15) is 10", lca_path.distance(5, 15) == 10)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all sparse_table tests passed")

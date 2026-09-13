"""Tests for LCA binary lifting: matches naive root-path intersection + BFS distance."""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lca import LCA, naive_lca  # noqa: E402


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
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def _random_tree(n, rng):
    """Random rooted tree: node i (>=1) attaches to a random earlier node. parent[0] = -1."""
    parent = [-1] * n
    edges = []
    for i in range(1, n):
        p = rng() % i
        parent[i] = p
        edges.append((p, i))
    return parent, edges


def _bfs_dist(n, edges, src):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def _naive_kth(v, k, parent):
    for _ in range(k):
        if v == -1:
            return -1
        v = parent[v]
    return v


def main():
    # ---- 1. LCA matches naive on many random trees, all pairs ---------------------------
    rng = _lcg(2024)
    mism_lca = mism_dist = mism_kth = 0
    for _ in range(120):
        n = 2 + rng() % 12
        parent, edges = _random_tree(n, rng)
        tree = LCA(n, edges=edges, root=0)
        dist0 = None
        for u in range(n):
            d = _bfs_dist(n, edges, u)
            for v in range(n):
                if tree.lca(u, v) != naive_lca(u, v, parent):
                    mism_lca += 1
                if tree.distance(u, v) != d[v]:
                    mism_dist += 1
        # k-th ancestor spot check
        for v in range(n):
            for k in range(0, n):
                if tree.kth_ancestor(v, k) != _naive_kth(v, k, parent):
                    mism_kth += 1
    check("LCA matches naive root-path intersection (120 trees)", mism_lca == 0,
          f"{mism_lca} mismatched")
    check("distance matches BFS shortest path", mism_dist == 0, f"{mism_dist} mismatched")
    check("kth_ancestor matches walking parents", mism_kth == 0, f"{mism_kth} mismatched")

    # ---- 2. LCA identities --------------------------------------------------------------
    rng = _lcg(55)
    ok_root = ok_self = ok_sym = True
    for _ in range(50):
        n = 3 + rng() % 10
        parent, edges = _random_tree(n, rng)
        tree = LCA(n, edges=edges, root=0)
        for u in range(n):
            if tree.lca(u, 0) != 0:
                ok_root = False
            if tree.lca(u, u) != u:
                ok_self = False
            for v in range(n):
                if tree.lca(u, v) != tree.lca(v, u):
                    ok_sym = False
    check("lca(u, root) == root", ok_root)
    check("lca(u, u) == u", ok_self)
    check("lca symmetric", ok_sym)

    # ---- 3. is_ancestor ------------------------------------------------------------------
    # tree: 0 -> 1 -> 3, 0 -> 2
    edges = [(0, 1), (1, 3), (0, 2)]
    tree = LCA(4, edges=edges, root=0)
    check("0 is ancestor of 3", tree.is_ancestor(0, 3))
    check("1 is ancestor of 3", tree.is_ancestor(1, 3))
    check("2 is NOT ancestor of 3", not tree.is_ancestor(2, 3))
    check("3 is NOT ancestor of 1", not tree.is_ancestor(3, 1))
    check("node is its own ancestor", tree.is_ancestor(3, 3))

    # ---- 4. jump along path -------------------------------------------------------------
    # path 3 - 1 - 0 - 2, so jump(3,2,*) walks 3,1,0,2
    check("jump 0 steps = u", tree.jump(3, 2, 0) == 3)
    check("jump 1 step toward 2 = 1", tree.jump(3, 2, 1) == 1)
    check("jump 2 steps toward 2 = 0 (the LCA)", tree.jump(3, 2, 2) == 0)
    check("jump 3 steps toward 2 = 2", tree.jump(3, 2, 3) == 2)
    check("jump past end = -1", tree.jump(3, 2, 4) == -1)
    check("distance(3,2) = 3", tree.distance(3, 2) == 3)

    # ---- 5. build from parent array matches build from edges ----------------------------
    rng = _lcg(321)
    ok = True
    for _ in range(40):
        n = 2 + rng() % 12
        parent, edges = _random_tree(n, rng)
        t_edges = LCA(n, edges=edges, root=0)
        t_par = LCA(n, parent=parent, root=0)
        for u in range(n):
            for v in range(n):
                if t_edges.lca(u, v) != t_par.lca(u, v):
                    ok = False
            if t_edges.depth[u] != t_par.depth[u]:
                ok = False
    check("parent-array build == edge-list build", ok)

    # ---- 6. edge cases ------------------------------------------------------------------
    single = LCA(1, edges=[], root=0)
    check("single node lca(0,0)=0", single.lca(0, 0) == 0)
    check("single node distance 0", single.distance(0, 0) == 0)
    check("kth_ancestor past root = -1", tree.kth_ancestor(3, 10) == -1)
    try:
        LCA(0, edges=[])
        check("n=0 raises", False)
    except ValueError:
        check("n=0 raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

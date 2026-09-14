"""Tests for Euler tour: contiguous subtree intervals, ancestor test, subtree sums vs brute force."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import euler_tour as ET  # noqa: E402


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


def main():
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    t = ET.euler_tour(6, edges, root=0)

    # ---- 1. entry times are a permutation of 0..n-1 -------------------------------------
    check("entry times are a permutation", sorted(t["tin"]) == list(range(6)))

    # ---- 2. subtree sizes are correct ---------------------------------------------------
    check("root subtree covers all n", ET.subtree_size(t, 0) == 6)
    check("subtree of 1 has size 3", ET.subtree_size(t, 1) == 3)
    check("leaf subtree has size 1", ET.subtree_size(t, 3) == 1)

    # ---- 3. subtree is a contiguous tour interval of exactly its size -------------------
    ok = True
    for v in range(6):
        lo, hi = ET.subtree_range(t, v)
        if hi - lo + 1 != ET.subtree_size(t, v):
            ok = False
        if len(ET.subtree_nodes(t, v)) != ET.subtree_size(t, v):
            ok = False
    check("every subtree is a contiguous interval of its size", ok)

    # ---- 4. ancestor test on the known tree ---------------------------------------------
    check("0 is ancestor of everyone", all(ET.is_ancestor(t, 0, v) for v in range(6)))
    check("1 is ancestor of 3 and 4", ET.is_ancestor(t, 1, 3) and ET.is_ancestor(t, 1, 4))
    check("2 is not an ancestor of 4", not ET.is_ancestor(t, 2, 4))
    check("a node is its own ancestor", ET.is_ancestor(t, 3, 3))

    # ---- 5. subtree sum matches a direct sum --------------------------------------------
    vals = [10, 20, 30, 40, 50, 60]
    check("subtree_sum(1) == 20+40+50", abs(ET.subtree_sum(t, vals, 1) - 110) < 1e-9)
    check("subtree_sum(root) == total", abs(ET.subtree_sum(t, vals, 0) - sum(vals)) < 1e-9)
    check("subtree_sum(leaf) == its value", abs(ET.subtree_sum(t, vals, 5) - 60) < 1e-9)

    # ---- 6. subtree_nodes matches brute force -------------------------------------------
    ok = all(set(ET.subtree_nodes(t, v)) == ET.brute_subtree_nodes(6, edges, v, 0) for v in range(6))
    check("subtree_nodes matches brute force", ok)

    # ---- 7. path graph: subtree sizes are n, n-1, ..., 1 --------------------------------
    n = 7
    path = [(i, i + 1) for i in range(n - 1)]
    tp = ET.euler_tour(n, path, root=0)
    check("path subtree sizes are n..1",
          [ET.subtree_size(tp, v) for v in range(n)] == list(range(n, 0, -1)))
    check("path: every node is an ancestor of all later ones",
          all(ET.is_ancestor(tp, u, v) for u in range(n) for v in range(u, n)))

    # ---- 8. star: root subtree is n, every leaf subtree is 1 ----------------------------
    star = [(0, i) for i in range(1, 6)]
    ts = ET.euler_tour(6, star, root=0)
    check("star root subtree == n", ET.subtree_size(ts, 0) == 6)
    check("star leaves have subtree size 1", all(ET.subtree_size(ts, v) == 1 for v in range(1, 6)))
    check("star leaves are not ancestors of each other",
          not ET.is_ancestor(ts, 1, 2) and not ET.is_ancestor(ts, 2, 1))

    # ---- 9. depth and parent are consistent ---------------------------------------------
    check("root has depth 0 and no parent", tp["depth"][0] == 0 and tp["parent"][0] == -1)
    check("path depths are 0..n-1", tp["depth"] == list(range(n)))

    # ---- 10. randomized cross-check on subtree nodes and ancestor tests -----------------
    rnd = _lcg(7)
    mism = 0
    for _ in range(60):
        m = 8 + int(rnd() * 10)
        redges = [(i, int(rnd() * i)) for i in range(1, m)]
        tt = ET.euler_tour(m, redges, 0)
        for v in range(m):
            if set(ET.subtree_nodes(tt, v)) != ET.brute_subtree_nodes(m, redges, v, 0):
                mism += 1
        for u in range(m):
            su = ET.brute_subtree_nodes(m, redges, u, 0)
            for v in range(m):
                if ET.is_ancestor(tt, u, v) != (v in su):
                    mism += 1
        if sorted(tt["tin"]) != list(range(m)):
            mism += 1
    check("randomized: subtree + ancestor + permutation all consistent", mism == 0, f"{mism}")

    # ---- 11. subtree sums on a random tree match brute --------------------------------
    rnd = _lcg(99)
    m = 15
    redges = [(i, int(rnd() * i)) for i in range(1, m)]
    tt = ET.euler_tour(m, redges, 0)
    vals = [rnd() * 10 for _ in range(m)]
    ok = True
    for v in range(m):
        brute = sum(vals[node] for node in ET.brute_subtree_nodes(m, redges, v, 0))
        if abs(ET.subtree_sum(tt, vals, v) - brute) > 1e-7:
            ok = False
    check("subtree sums match brute on a random tree", ok)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

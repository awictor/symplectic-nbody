"""Tests for vp_tree: pruned search equals brute force across metrics, pruning fires, range queries."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vp_tree import (VPTree, euclidean, manhattan, edit_distance, angular_distance,  # noqa: E402
                     brute_k_nearest, brute_within)


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

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def randint(self, n):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) % n


def same_result(a, b, tol=1e-9):
    """Compare two (dist, idx, item) result lists by distance (ties may reorder indices)."""
    if len(a) != len(b):
        return False
    return all(abs(x[0] - y[0]) < tol for x, y in zip(a, b))


def main():
    rng = LCG(2024)

    # ---- 1. Euclidean points: k-NN matches brute force over many queries --------------
    pts = [(rng.u() * 100, rng.u() * 100, rng.u() * 100) for _ in range(400)]
    tree = VPTree(pts, euclidean, seed=7)
    mism = 0
    for _ in range(200):
        q = (rng.u() * 100, rng.u() * 100, rng.u() * 100)
        for k in (1, 5, 10):
            got = tree.k_nearest(q, k)
            ref = brute_k_nearest(pts, q, k, euclidean)
            if not same_result(got, ref):
                mism += 1
    check("Euclidean k-NN matches brute force (200 queries x 3 k)", mism == 0, f"{mism} mismatches")

    # ---- 2. pruning actually fires: far fewer distance calls than n -------------------
    q = (50.0, 50.0, 50.0)
    tree.nearest(q)
    calls = tree.distance_calls
    check("nearest visits far fewer than all n nodes", calls < len(pts),
          f"{calls} calls vs n={len(pts)}")

    # ---- 3. querying an indexed point returns itself at distance 0 --------------------
    idx = 123
    res = tree.nearest(pts[idx])
    check("query on an indexed point returns distance 0", abs(res[0]) < 1e-12, f"d={res[0]}")

    # ---- 4. Manhattan metric --------------------------------------------------------
    tman = VPTree(pts, manhattan, seed=3)
    mism_m = 0
    for _ in range(80):
        q = (rng.u() * 100, rng.u() * 100, rng.u() * 100)
        if not same_result(tman.k_nearest(q, 5), brute_k_nearest(pts, q, 5, manhattan)):
            mism_m += 1
    check("Manhattan k-NN matches brute force", mism_m == 0, f"{mism_m} mismatches")

    # ---- 5. string edit distance (a non-Euclidean metric with no coordinates) --------
    alpha = "abcde"
    words = []
    for _ in range(300):
        L = 3 + rng.randint(6)
        words.append("".join(alpha[rng.randint(len(alpha))] for _ in range(L)))
    wtree = VPTree(words, edit_distance, seed=11)
    mism_e = 0
    for _ in range(120):
        L = 3 + rng.randint(6)
        q = "".join(alpha[rng.randint(len(alpha))] for _ in range(L))
        got = wtree.k_nearest(q, 3)
        ref = brute_k_nearest(words, q, 3, edit_distance)
        if not same_result(got, ref):
            mism_e += 1
    check("edit-distance k-NN matches brute force", mism_e == 0, f"{mism_e} mismatches")

    # ---- 6. cosine distance ---------------------------------------------------------
    vecs = [tuple(rng.u() - 0.5 for _ in range(5)) for _ in range(250)]
    ctree = VPTree(vecs, angular_distance, seed=5)
    mism_c = 0
    for _ in range(80):
        q = tuple(rng.u() - 0.5 for _ in range(5))
        if not same_result(ctree.k_nearest(q, 4), brute_k_nearest(vecs, q, 4, angular_distance)):
            mism_c += 1
    check("angular-distance k-NN matches brute force", mism_c == 0, f"{mism_c} mismatches")

    # ---- 7. range queries exactly match brute force ----------------------------------
    mism_r = 0
    for _ in range(100):
        q = (rng.u() * 100, rng.u() * 100, rng.u() * 100)
        radius = 15.0 + rng.u() * 20.0
        got = tree.within(q, radius)
        ref = brute_within(pts, q, radius, euclidean)
        if not same_result(got, ref):
            mism_r += 1
    check("range queries match brute force", mism_r == 0, f"{mism_r} mismatches")

    # ---- 8. returned distances are correct and sorted --------------------------------
    q = (30.0, 40.0, 50.0)
    res = tree.k_nearest(q, 8)
    dists_ok = all(abs(d - euclidean(q, it)) < 1e-9 for d, i, it in res)
    sorted_ok = all(res[i][0] <= res[i + 1][0] for i in range(len(res) - 1))
    check("returned distances correct", dists_ok)
    check("results sorted by distance", sorted_ok)
    # index/item consistency
    check("returned index matches item", all(pts[i] == it for d, i, it in res))

    # ---- 9. edge cases ---------------------------------------------------------------
    empty = VPTree([], euclidean)
    check("empty tree nearest is None", empty.nearest((0, 0)) is None)
    check("empty tree k_nearest is []", empty.k_nearest((0, 0), 3) == [])
    single = VPTree([(1.0, 2.0)], euclidean)
    check("single-point tree returns that point", single.nearest((5.0, 5.0))[1] == 0)
    # k larger than n
    small = VPTree([(0, 0), (1, 1), (2, 2)], euclidean)
    check("k > n returns all points", len(small.k_nearest((0, 0), 10)) == 3)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

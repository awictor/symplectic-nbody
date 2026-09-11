"""Tests for kdtree.py -- k-d tree nearest-neighbour search.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Every query is cross-checked
against a brute-force linear scan over deterministic pseudo-random point clouds.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import kdtree as K  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def lcg_points(seed, n, dim, hi=100.0):
    """Deterministic pseudo-random points via an LCG (high bits)."""
    state = seed
    pts = []
    for _ in range(n):
        coord = []
        for _ in range(dim):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            coord.append((state >> 8) / (1 << 24) * hi)
        pts.append(tuple(coord))
    return pts


# --- construction -----------------------------------------------------------
pts = lcg_points(1, 500, 2)
tree = K.KDTree(pts)
check("tree records its size", tree.size == 500)
check("tree infers the dimension", tree.k == 2)
check("a balanced build is ~log2(n) tall", tree.height() <= 2 * math.ceil(math.log2(500)))
check("empty tree has no nearest", K.KDTree([]).nearest((0, 0)) is None)
check("empty tree k_nearest is empty", K.KDTree([]).k_nearest((0, 0), 3) == [])
check("single-point tree returns that point", K.KDTree([(1, 2)]).nearest((9, 9)) == (1, 2))
try:
    K.KDTree([(1, 2), (3, 4, 5)])
    check("rejects mixed-dimension points", False)
except ValueError:
    check("rejects mixed-dimension points", True)

# --- nearest neighbour matches brute force ---------------------------------
queries = lcg_points(99, 300, 2)
nearest_ok = all(tree.nearest(q) == K.brute_nearest(pts, q) for q in queries)
check("nearest matches brute force over 300 queries (2D)", nearest_ok)
# a query exactly on a stored point returns that point
check("a query at a stored point returns it", tree.nearest(pts[42]) == pts[42])

# --- k-nearest matches brute force -----------------------------------------
knn_ok = True
for q in queries[:100]:
    for k in (1, 3, 7):
        if tree.k_nearest(q, k) != K.brute_k_nearest(pts, q, k):
            knn_ok = False
            break
check("k-nearest matches brute force for k=1,3,7", knn_ok)
check("k_nearest(1) equals nearest", tree.k_nearest((50, 50), 1)[0] == tree.nearest((50, 50)))
check("k_nearest is sorted nearest-first",
      all(K._dist2(a, (50, 50)) <= K._dist2(b, (50, 50))
          for a, b in zip(tree.k_nearest((50, 50), 10), tree.k_nearest((50, 50), 10)[1:])))
check("k larger than n returns all points", len(tree.k_nearest((0, 0), 999)) == 500)
check("k_nearest(0) is empty", tree.k_nearest((0, 0), 0) == [])

# --- radius query matches brute force --------------------------------------
radius_ok = True
for q in queries[:100]:
    for r in (5.0, 15.0, 40.0):
        got = sorted(tree.within_radius(q, r))
        exp = sorted(K.brute_within_radius(pts, q, r))
        if got != exp:
            radius_ok = False
            break
check("radius query matches brute force for r=5,15,40", radius_ok)
check("radius 0 returns only exact matches",
      tree.within_radius(pts[10], 0.0) == [pts[10]])
check("huge radius returns every point", len(tree.within_radius((50, 50), 1e9)) == 500)
# every returned point is genuinely within the radius
within = tree.within_radius((50, 50), 20.0)
check("all radius results are truly within the radius",
      all(math.dist(p, (50, 50)) <= 20.0 + 1e-9 for p in within))

# --- 3D correctness ---------------------------------------------------------
pts3 = lcg_points(7, 400, 3, hi=10.0)
tree3 = K.KDTree(pts3)
q3 = lcg_points(8, 100, 3, hi=10.0)
check("nearest matches brute force in 3D", all(tree3.nearest(q) == K.brute_nearest(pts3, q) for q in q3))
check("3D k-nearest matches brute force",
      all(tree3.k_nearest(q, 4) == K.brute_k_nearest(pts3, q, 4) for q in q3[:40]))

# --- clustered / duplicate points ------------------------------------------
dup = [(1.0, 1.0)] * 5 + [(2.0, 2.0), (3.0, 3.0)]
dtree = K.KDTree(dup)
check("handles duplicate points", dtree.nearest((1.1, 1.1)) == (1.0, 1.0))
check("radius query counts duplicates", len(dtree.within_radius((1.0, 1.0), 0.01)) == 5)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall kdtree tests passed")

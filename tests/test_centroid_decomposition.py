"""Tests for centroid_decomposition: distance-pair counting + centroid tree vs brute force."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from centroid_decomposition import (build_centroid_tree, count_pairs_within_distance,
                                    centroid_tree_depth, brute_count_pairs_within_distance,
                                    all_pairs_distances)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


def random_tree(rng, n):
    return [(rng.randint(0, v - 1), v) for v in range(1, n)]


def is_valid_tree(n, parent, root):
    """The centroid tree is a valid rooted tree: one root, every other vertex reaches it acyclically."""
    if root == -1:
        return n == 0
    if parent[root] != -1:
        return False
    for v in range(n):
        if v == root:
            continue
        steps = 0
        u = v
        while u != root:
            if parent[u] == -1:
                return False
            u = parent[u]
            steps += 1
            if steps > n:
                return False
    return True


# --- known distance-pair counts --------------------------------------------
path = [(0, 1), (1, 2), (2, 3), (3, 4)]
check("path k=1: 4 adjacent pairs", count_pairs_within_distance(5, path, 1) == 4)
check("path k=2: 7 pairs", count_pairs_within_distance(5, path, 2) == 7)
check("path k=4: all 10 pairs", count_pairs_within_distance(5, path, 4) == 10)

star = [(0, 1), (0, 2), (0, 3), (0, 4)]
check("star k=1: 4 hub-leaf pairs", count_pairs_within_distance(5, star, 1) == 4)
check("star k=2: all 10 pairs (leaves are distance 2 apart)",
      count_pairs_within_distance(5, star, 2) == 10)

check("single vertex: 0 pairs", count_pairs_within_distance(1, [], 5) == 0)
check("k=0: 0 pairs", count_pairs_within_distance(5, path, 0) == 0)

# --- vs brute for every k on random trees ----------------------------------
rng = LCG(2026)
count_ok = True
for _ in range(300):
    n = rng.randint(1, 30)
    edges = random_tree(rng, n)
    for k in range(0, n + 1):
        if count_pairs_within_distance(n, edges, k) != brute_count_pairs_within_distance(n, edges, k):
            count_ok = False
            print(f"  mismatch: n={n} k={k} edges={edges}")
            break
    if not count_ok:
        break
check("distance-pair count matches brute force for every k (300 random trees)", count_ok)

# --- centroid tree is a valid tree -----------------------------------------
rng = LCG(4242)
tree_ok = True
for _ in range(300):
    n = rng.randint(1, 40)
    edges = random_tree(rng, n)
    parent, root = build_centroid_tree(n, edges)
    if not is_valid_tree(n, parent, root):
        tree_ok = False
        print(f"  invalid centroid tree: n={n} edges={edges}")
        break
check("the centroid tree is a valid rooted tree (300 random trees)", tree_ok)

# --- every vertex appears exactly once in the centroid tree ----------------
rng = LCG(777)
cover_ok = True
for _ in range(300):
    n = rng.randint(1, 40)
    edges = random_tree(rng, n)
    parent, root = build_centroid_tree(n, edges)
    # exactly one root, all others have a parent
    roots = [v for v in range(n) if parent[v] == -1]
    if len(roots) != 1 or roots[0] != root:
        cover_ok = False
        break
check("centroid tree has exactly one root, all vertices covered", cover_ok)

# --- depth is logarithmic --------------------------------------------------
rng = LCG(555)
depth_ok = True
import math
for _ in range(200):
    n = rng.randint(1, 200)
    edges = random_tree(rng, n)
    d = centroid_tree_depth(n, edges)
    # centroid tree depth is at most ~log2(n); allow a small constant slack
    if n >= 2 and d > 2 * math.log2(n) + 2:
        depth_ok = False
        print(f"  depth {d} too large for n={n}")
        break
check("centroid tree depth is O(log n) (200 trees up to n=200)", depth_ok)

# a path is the worst case for centroid depth but still logarithmic
check("path on 63 vertices has centroid depth <= 6",
      centroid_tree_depth(63, [(i, i + 1) for i in range(62)]) <= 6)

# --- balanced binary tree known distances ----------------------------------
# complete binary tree of 7 nodes: 0 root, children 2i+1, 2i+2
bt = []
for i in range(3):
    bt.append((i, 2 * i + 1))
    bt.append((i, 2 * i + 2))
# all pairs at distance <= 2
brute = brute_count_pairs_within_distance(7, bt, 2)
check("balanced binary tree (7 nodes) k=2 matches brute",
      count_pairs_within_distance(7, bt, 2) == brute)

# --- large tree solves and matches brute for a few k -----------------------
rng = LCG(31337)
n = 400
edges = random_tree(rng, n)
big_ok = True
for k in (1, 5, 20, n):
    if count_pairs_within_distance(n, edges, k) != brute_count_pairs_within_distance(n, edges, k):
        big_ok = False
        break
check("400-vertex tree: pair counts match brute for several k", big_ok)

# --- total pairs at k = n-1 is C(n,2) --------------------------------------
rng = LCG(99)
total_ok = True
for _ in range(100):
    n = rng.randint(2, 20)
    edges = random_tree(rng, n)
    if count_pairs_within_distance(n, edges, n - 1) != n * (n - 1) // 2:
        total_ok = False
        break
check("at k = n-1 (diameter bound) all C(n,2) pairs are counted", total_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all centroid_decomposition tests passed")

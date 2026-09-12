"""Tests for tree_isomorphism: AHU canonical form + isomorphism vs brute permutation check."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tree_isomorphism import (rooted_canonical, centers, unrooted_canonical,
                              rooted_isomorphic, isomorphic, brute_isomorphic)

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
    """A random labelled tree on n vertices: attach each new vertex to a random earlier one."""
    edges = []
    for v in range(1, n):
        u = rng.randint(0, v - 1)
        edges.append((u, v))
    return edges


def relabel(n, edges, perm):
    return [(perm[u], perm[v]) for u, v in edges]


def random_perm(rng, n):
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.rand() % (i + 1)
        p[i], p[j] = p[j], p[i]
    return p


# --- known cases ------------------------------------------------------------
path4 = [(0, 1), (1, 2), (2, 3)]
star4 = [(0, 1), (0, 2), (0, 3)]
check("relabelled path is isomorphic to itself",
      isomorphic(4, path4, 4, [(3, 2), (2, 1), (1, 0)]))
check("path and star (same size) are NOT isomorphic", not isomorphic(4, path4, 4, star4))
check("single vertex trees are isomorphic", isomorphic(1, [], 1, []))
check("different sizes are not isomorphic", not isomorphic(3, [(0, 1), (1, 2)], 4, path4))

# --- centers ----------------------------------------------------------------
check("odd path has one center", centers(5, [(0, 1), (1, 2), (2, 3), (3, 4)]) == [2])
check("even path has two centers", sorted(centers(4, path4)) == [1, 2])
check("star center is the hub", centers(4, star4) == [0])
check("single vertex is its own center", centers(1, []) == [0])

# --- rooted canonical form --------------------------------------------------
# a leaf is "()", and children order must not matter
c1 = rooted_canonical(3, [(0, 1), (0, 2)], 0)
c2 = rooted_canonical(3, [(0, 2), (0, 1)], 0)
check("rooted canonical is independent of child order", c1 == c2)
check("a single vertex canonical form is '()'", rooted_canonical(1, [], 0) == "()")

# rooting a path at its end vs its middle gives different rooted forms
end = rooted_canonical(3, [(0, 1), (1, 2)], 0)
mid = rooted_canonical(3, [(0, 1), (1, 2)], 1)
check("rooting at different vertices can give different rooted forms", end != mid)

# --- relabelled trees are always isomorphic (vs brute) ---------------------
rng = LCG(2026)
relabel_ok = True
for _ in range(300):
    n = rng.randint(1, 9)
    edges = random_tree(rng, n)
    perm = random_perm(rng, n)
    edges2 = relabel(n, edges, perm)
    if not isomorphic(n, edges, n, edges2):
        relabel_ok = False
        print(f"  relabelled copy not iso: n={n} edges={edges} perm={perm}")
        break
check("a relabelled copy always tests isomorphic (300 random trees)", relabel_ok)

# --- full agreement with brute force ---------------------------------------
rng = LCG(4242)
agree_ok = True
saw_iso = saw_noniso = False
for _ in range(400):
    n = rng.randint(1, 8)
    e1 = random_tree(rng, n)
    # half the time make an isomorphic copy, half the time an independent random tree
    if rng.rand() % 2 == 0:
        e2 = relabel(n, e1, random_perm(rng, n))
    else:
        e2 = random_tree(rng, n)
    fast = isomorphic(n, e1, n, e2)
    brute = brute_isomorphic(n, e1, n, e2)
    if fast != brute:
        agree_ok = False
        print(f"  disagreement: fast={fast} brute={brute} n={n} e1={e1} e2={e2}")
        break
    if brute:
        saw_iso = True
    else:
        saw_noniso = True
check("AHU isomorphism matches brute-force permutation check (400 tree pairs)", agree_ok)
check("random suite saw both isomorphic and non-isomorphic pairs", saw_iso and saw_noniso)

# --- rooted isomorphism -----------------------------------------------------
# same tree rooted at isomorphic positions
t = [(0, 1), (0, 2), (1, 3), (1, 4)]
check("rooted iso: identical rooting is isomorphic", rooted_isomorphic(5, t, 0, 5, t, 0))
# a path rooted at an endpoint vs the middle should NOT be rooted-isomorphic
check("rooted iso: path rooted at end vs middle differ",
      not rooted_isomorphic(3, [(0, 1), (1, 2)], 0, 3, [(0, 1), (1, 2)], 1))

# --- two trees with the same degree sequence but different shape -----------
# "caterpillar" vs "spider" on 7 vertices, both with the same number of leaves
tA = [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5), (3, 6)]
tB = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6)]
check("same-size different-shape trees are not isomorphic (matches brute)",
      isomorphic(7, tA, 7, tB) == brute_isomorphic(7, tA, 7, tB))

# --- canonical form equality is exactly isomorphism ------------------------
rng = LCG(777)
canon_ok = True
for _ in range(200):
    n = rng.randint(1, 8)
    e1 = random_tree(rng, n)
    e2 = random_tree(rng, n)
    same_form = unrooted_canonical(n, e1) == unrooted_canonical(n, e2)
    if same_form != brute_isomorphic(n, e1, n, e2):
        canon_ok = False
        break
check("canonical-form equality coincides with isomorphism", canon_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all tree_isomorphism tests passed")

"""Tests for matrix_tree: spanning-tree counting via the Laplacian cofactor vs brute enumeration."""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matrix_tree import (count_spanning_trees, weighted_spanning_tree_sum,
                         brute_count_spanning_trees, laplacian, _det)

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


def complete(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def cycle(n):
    return [(i, (i + 1) % n) for i in range(n)]


# --- known values -----------------------------------------------------------
check("K3 (triangle) has 3 spanning trees", count_spanning_trees(3, complete(3)) == 3)
check("K4 has 16 spanning trees (Cayley 4^2)", count_spanning_trees(4, complete(4)) == 16)
check("K5 has 125 spanning trees (Cayley 5^3)", count_spanning_trees(5, complete(5)) == 125)
check("K6 has 1296 spanning trees (Cayley 6^4)", count_spanning_trees(6, complete(6)) == 1296)

check("cycle C5 has 5 spanning trees", count_spanning_trees(5, cycle(5)) == 5)
check("cycle C7 has 7 spanning trees", count_spanning_trees(7, cycle(7)) == 7)

check("a tree has exactly 1 spanning tree", count_spanning_trees(4, [(0, 1), (1, 2), (2, 3)]) == 1)
check("single vertex has 1 spanning tree", count_spanning_trees(1, []) == 1)
check("disconnected graph has 0 spanning trees", count_spanning_trees(4, [(0, 1), (2, 3)]) == 0)
check("edgeless multi-vertex graph has 0 spanning trees", count_spanning_trees(3, []) == 0)

# --- Cayley's formula for all small complete graphs ------------------------
cayley_ok = all(count_spanning_trees(n, complete(n)) == n ** (n - 2) for n in range(2, 8))
check("complete graph K_n has n^(n-2) spanning trees (n up to 7)", cayley_ok)

# --- vs brute force on random graphs ---------------------------------------
rng = LCG(2026)
brute_ok = True
saw_connected = saw_disconnected = False
for _ in range(400):
    n = rng.randint(1, 7)
    edges = []
    seen = set()
    for _ in range(rng.randint(0, n * 2)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and (min(u, v), max(u, v)) not in seen:
            seen.add((min(u, v), max(u, v)))
            edges.append((u, v))
    mt = count_spanning_trees(n, edges)
    bf = brute_count_spanning_trees(n, edges)
    if mt != bf:
        brute_ok = False
        print(f"  mismatch: matrix-tree={mt} brute={bf} n={n} edges={edges}")
        break
    if mt > 0:
        saw_connected = True
    else:
        saw_disconnected = True
check("Matrix-Tree count matches brute enumeration (400 random graphs)", brute_ok)
check("suite saw connected and disconnected graphs", saw_connected and saw_disconnected)

# --- any cofactor gives the same value (delete any row/column) -------------
rng = LCG(4242)
cofactor_ok = True
for _ in range(200):
    n = rng.randint(2, 7)
    edges = []
    seen = set()
    for _ in range(rng.randint(n, n * 2)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and (min(u, v), max(u, v)) not in seen:
            seen.add((min(u, v), max(u, v)))
            edges.append((u, v))
    L = laplacian(n, edges)
    # cofactor deleting row/col 0 vs deleting the last -> must match
    minor0 = [[L[i][j] for j in range(1, n)] for i in range(1, n)]
    minor_last = [[L[i][j] for j in range(n - 1)] for i in range(n - 1)]
    if _det(minor0) != _det(minor_last):
        cofactor_ok = False
        break
check("every cofactor of the Laplacian gives the same spanning-tree count", cofactor_ok)

# --- weighted spanning-tree sum --------------------------------------------
# triangle with edge weights a,b,c: spanning trees are the 3 pairs, sum = ab+bc+ca
a, b, c = 2, 3, 5
tri_w = [(0, 1, a), (1, 2, b), (2, 0, c)]
expected = a * b + b * c + c * a
check("weighted triangle spanning-tree sum is ab+bc+ca",
      weighted_spanning_tree_sum(3, tri_w) == Fraction(expected))

# unit weights reduce to the plain count
rng = LCG(777)
unit_ok = True
for _ in range(150):
    n = rng.randint(2, 6)
    edges = []
    seen = set()
    for _ in range(rng.randint(n, n * 2)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and (min(u, v), max(u, v)) not in seen:
            seen.add((min(u, v), max(u, v)))
            edges.append((u, v))
    w_edges = [(u, v, 1) for u, v in edges]
    if weighted_spanning_tree_sum(n, w_edges) != Fraction(count_spanning_trees(n, edges)):
        unit_ok = False
        break
check("weighted sum with unit weights equals the unweighted count", unit_ok)

# --- adding an edge never decreases the spanning-tree count ----------------
rng = LCG(31337)
monotone_ok = True
for _ in range(150):
    n = rng.randint(2, 7)
    edges = []
    seen = set()
    for _ in range(rng.randint(0, n)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and (min(u, v), max(u, v)) not in seen:
            seen.add((min(u, v), max(u, v)))
            edges.append((u, v))
    before = count_spanning_trees(n, edges)
    # add a new edge
    for _ in range(5):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and (min(u, v), max(u, v)) not in seen:
            edges2 = edges + [(u, v)]
            after = count_spanning_trees(n, edges2)
            if after < before:
                monotone_ok = False
            break
check("adding an edge never decreases the spanning-tree count", monotone_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all matrix_tree tests passed")

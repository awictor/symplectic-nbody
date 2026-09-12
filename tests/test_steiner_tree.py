"""Tests for steiner_tree: Dreyfus-Wagner bitmask DP vs brute-force Steiner-point enumeration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from steiner_tree import steiner_tree, brute_steiner_tree, _mst_weight_on

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


def connected_graph(rng, n):
    """A random connected weighted graph: a spanning path plus extra random edges."""
    edges = []
    verts = list(range(n))
    for i in range(1, n):
        j = rng.randint(0, i - 1)
        edges.append((j, i, rng.randint(1, 15)))
    for _ in range(rng.randint(0, n)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.append((u, v, rng.randint(1, 15)))
    return edges


# --- known cases ------------------------------------------------------------
# 4 corners of a square, cheaper to route through a center vertex
edges = [(0, 1, 10), (1, 2, 10), (2, 3, 10), (3, 0, 10),
         (0, 4, 3), (1, 4, 3), (2, 4, 3), (3, 4, 3)]
check("4 corners connect through the center for cost 12", steiner_tree(5, edges, [0, 1, 2, 3]) == 12)

# two terminals: answer is the shortest path
edges = [(0, 1, 5), (1, 2, 5), (0, 2, 20)]
check("two terminals -> shortest path (10, not the direct edge 20)",
      steiner_tree(3, edges, [0, 2]) == 10)

check("single terminal costs 0", steiner_tree(3, edges, [1]) == 0)
check("no terminals costs 0", steiner_tree(3, edges, []) == 0)

# disconnected terminals -> inf
check("unreachable terminals -> inf", steiner_tree(3, [(0, 1, 1)], [0, 2]) == float("inf"))

# --- all-vertices-terminal reduces to the MST ------------------------------
rng = LCG(2026)
mst_ok = True
for _ in range(150):
    n = rng.randint(2, 8)
    edges = connected_graph(rng, n)
    st = steiner_tree(n, edges, list(range(n)))
    mst = _mst_weight_on(set(range(n)), edges)
    if st != mst:
        mst_ok = False
        print(f"  Steiner(all)={st} MST={mst} n={n} edges={edges}")
        break
check("all vertices terminal: Steiner tree equals the MST", mst_ok)

# --- exhaustive validation vs brute ----------------------------------------
rng = LCG(4242)
brute_ok = True
saw_steiner_point = False
for _ in range(400):
    n = rng.randint(2, 8)
    edges = connected_graph(rng, n)
    k = rng.randint(2, min(4, n))
    # pick k distinct terminals
    terms = []
    pool = list(range(n))
    for _ in range(k):
        idx = rng.rand() % len(pool)
        terms.append(pool.pop(idx))
    st = steiner_tree(n, edges, terms)
    bf = brute_steiner_tree(n, edges, terms)
    if st != bf:
        brute_ok = False
        print(f"  mismatch: dp={st} brute={bf} n={n} terms={terms} edges={edges}")
        break
    # detect a case where using a Steiner point beats the terminal-only MST
    term_mst = _mst_weight_on(set(terms), edges)
    if st < (term_mst if term_mst is not None else float("inf")):
        saw_steiner_point = True
check("Steiner tree matches brute force (400 random graphs)", brute_ok)
check("random suite includes cases where a Steiner point strictly helps", saw_steiner_point)

# --- monotonicity: more terminals never costs less -------------------------
rng = LCG(777)
monotone_ok = True
for _ in range(100):
    n = rng.randint(3, 8)
    edges = connected_graph(rng, n)
    w2 = steiner_tree(n, edges, [0, 1])
    w3 = steiner_tree(n, edges, [0, 1, 2])
    if w3 < w2:
        monotone_ok = False
        break
check("adding a terminal never decreases the Steiner tree weight", monotone_ok)

# --- terminal order does not matter ----------------------------------------
edges = [(0, 1, 10), (1, 2, 10), (2, 3, 10), (3, 0, 10), (0, 4, 3), (1, 4, 3), (2, 4, 3), (3, 4, 3)]
check("result is independent of terminal ordering",
      steiner_tree(5, edges, [0, 1, 2, 3]) == steiner_tree(5, edges, [3, 1, 0, 2]))

# --- a path graph: Steiner tree of the endpoints is the whole path ---------
edges = [(i, i + 1, 2) for i in range(6)]
check("path graph: connecting the two ends uses the whole path",
      steiner_tree(7, edges, [0, 6]) == 12)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all steiner_tree tests passed")

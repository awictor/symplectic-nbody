"""Tests for arborescence: Chu-Liu/Edmonds minimum spanning arborescence vs brute force."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arborescence import min_arborescence, brute_min_arborescence, _is_arborescence

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


def valid_arborescence(n, parent, root, edges_set):
    """parent maps v -> chosen (u,v,w). Check it is a spanning arborescence using real edges."""
    for v in range(n):
        if v == root:
            if parent[v] is not None:
                return False
            continue
        if parent[v] is None:
            return False
        if (parent[v][0], parent[v][1], parent[v][2]) not in edges_set:
            return False
        if parent[v][1] != v:
            return False
    # every vertex reaches the root
    pmap = {v: parent[v][0] for v in range(n) if v != root}
    return _is_arborescence(n, pmap, root)


# --- known cases ------------------------------------------------------------
edges = [(0, 1, 1), (0, 2, 5), (1, 2, 2)]
w, parent = min_arborescence(3, edges, 0)
check("simple graph: min arborescence weight 3", w == 3)
check("simple graph: chosen edges form a valid arborescence",
      valid_arborescence(3, parent, 0, set(edges)))

# a cycle that must be contracted
edges = [(0, 1, 10), (1, 2, 1), (2, 1, 1), (0, 2, 10)]
w, _ = min_arborescence(3, edges, 0)
check("cycle-contraction case: weight 11", w == 11)

# unreachable vertex -> no arborescence
check("unreachable vertex: no arborescence", min_arborescence(3, [(0, 1, 1)], 0) == (None, None))

# single vertex (just the root)
check("single-vertex graph: weight 0", min_arborescence(1, [], 0) == (0, {0: None}))

# a plain out-tree: unique arborescence equals the sum of all edges
edges = [(0, 1, 3), (0, 2, 4), (1, 3, 5)]
w, _ = min_arborescence(4, edges, 0)
check("tree graph: weight is the sum of its edges", w == 12)

# --- exhaustive validation vs brute ----------------------------------------
rng = LCG(2026)
weight_ok = valid_ok = True
saw_feasible = saw_infeasible = False
for _ in range(500):
    n = rng.randint(1, 6)
    root = rng.randint(0, n - 1)
    m = rng.randint(0, n * 2)
    edges = []
    for _ in range(m):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.append((u, v, rng.randint(1, 20)))
    w, parent = min_arborescence(n, edges, root)
    bw, _ = brute_min_arborescence(n, edges, root)
    # weights must agree (both None, or equal)
    if (w is None) != (bw is None):
        weight_ok = False
        print(f"  feasibility mismatch: fast={w} brute={bw} n={n} root={root} edges={edges}")
        break
    if w is not None:
        saw_feasible = True
        if w != bw:
            weight_ok = False
            print(f"  weight mismatch: fast={w} brute={bw} edges={edges} root={root}")
            break
        if not valid_arborescence(n, parent, root, set(edges)):
            valid_ok = False
            print(f"  invalid arborescence returned: {parent}")
            break
    else:
        saw_infeasible = True
check("min arborescence weight matches brute force (500 random graphs)", weight_ok)
check("returned arborescence is always valid and uses real edges", valid_ok)
check("random suite hit both feasible and infeasible graphs", saw_feasible and saw_infeasible)

# --- multi-edges: cheapest parallel edge is used ---------------------------
edges = [(0, 1, 9), (0, 1, 2), (0, 1, 7)]
w, parent = min_arborescence(2, edges, 0)
check("multi-edges: the cheapest parallel edge is chosen", w == 2 and parent[1][2] == 2)

# --- nested cycles (two levels of contraction) -----------------------------
# root 0; a 3-cycle 1->2->3->1 all cheap, entered expensively from 0
edges = [(0, 1, 100), (1, 2, 1), (2, 3, 1), (3, 1, 1), (0, 2, 50), (2, 1, 3)]
w, parent = min_arborescence(4, edges, 0)
bw, _ = brute_min_arborescence(4, edges, 0)
check("nested/expensive-entry cycle matches brute", w == bw)
check("nested-cycle arborescence is valid", valid_arborescence(4, parent, 0, set(edges)))

# --- adding a cheaper edge never increases the optimum ---------------------
rng = LCG(4242)
monotone_ok = True
for _ in range(100):
    n = rng.randint(2, 6)
    root = 0
    edges = []
    # ensure feasibility: a base out-tree
    for v in range(1, n):
        edges.append((rng.randint(0, v - 1), v, rng.randint(5, 20)))
    w0, _ = min_arborescence(n, edges, root)
    # add a random (possibly cheaper) edge
    u = rng.randint(0, n - 1)
    v = rng.randint(1, n - 1)
    if u != v:
        edges2 = edges + [(u, v, rng.randint(1, 4))]
        w1, _ = min_arborescence(n, edges2, root)
        if w1 is not None and w0 is not None and w1 > w0:
            monotone_ok = False
            break
check("adding an edge never increases the minimum arborescence weight", monotone_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all arborescence tests passed")

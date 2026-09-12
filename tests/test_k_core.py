"""Tests for k_core: coreness, degeneracy, k-cores/shells vs brute-force peeling."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from k_core import (coreness, degeneracy, degeneracy_ordering, k_core, k_shell, shells,
                    brute_k_core, brute_coreness, _adj)

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


def random_graph(rng, n, p_num, p_den):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.rand() % p_den < p_num:
                edges.append((u, v))
    return edges


# --- known cases ------------------------------------------------------------
# triangle 0-1-2 with a pendant 3 on 0
tri_pend = [(0, 1), (1, 2), (2, 0), (0, 3)]
check("triangle+pendant coreness is [2,2,2,1]", coreness(4, tri_pend) == [2, 2, 2, 1])
check("triangle+pendant degeneracy is 2", degeneracy(4, tri_pend) == 2)
check("its 2-core is the triangle", k_core(4, tri_pend, 2) == [0, 1, 2])
check("its 1-shell is the pendant", k_shell(4, tri_pend, 1) == [3])

# K_n has coreness n-1 everywhere
for nn in (3, 4, 5):
    kn = [(i, j) for i in range(nn) for j in range(i + 1, nn)]
    check(f"K{nn} has coreness {nn-1} for all vertices",
          coreness(nn, kn) == [nn - 1] * nn)

# a path has degeneracy 1
check("path graph has degeneracy 1", degeneracy(6, [(i, i + 1) for i in range(5)]) == 1)
# a cycle has degeneracy 2
check("cycle graph has degeneracy 2", degeneracy(6, [(i, (i + 1) % 6) for i in range(6)]) == 2)

check("edgeless graph: all coreness 0", coreness(4, []) == [0, 0, 0, 0])
check("empty graph degeneracy 0", degeneracy(0, []) == 0)

# --- coreness matches brute definition -------------------------------------
rng = LCG(2026)
core_ok = True
for _ in range(400):
    n = rng.randint(1, 10)
    edges = random_graph(rng, n, rng.randint(1, 3), 4)
    if coreness(n, edges) != brute_coreness(n, edges):
        core_ok = False
        print(f"  coreness mismatch: n={n} edges={edges}")
        break
check("coreness matches brute-force peeling (400 random graphs)", core_ok)

# --- k-core vertex sets match brute for every k ----------------------------
rng = LCG(4242)
core_set_ok = True
for _ in range(300):
    n = rng.randint(1, 10)
    edges = random_graph(rng, n, rng.randint(1, 3), 4)
    deg = degeneracy(n, edges)
    for k in range(0, deg + 2):
        if k_core(n, edges, k) != brute_k_core(n, edges, k):
            core_set_ok = False
            print(f"  {k}-core mismatch: n={n} edges={edges}")
            break
    if not core_set_ok:
        break
check("k-core vertex sets match brute force for every k (300 graphs)", core_set_ok)

# --- k-core is nested: (k+1)-core is a subset of k-core --------------------
rng = LCG(777)
nested_ok = True
for _ in range(200):
    n = rng.randint(1, 10)
    edges = random_graph(rng, n, 2, 4)
    deg = degeneracy(n, edges)
    for k in range(0, deg + 1):
        a = set(k_core(n, edges, k))
        b = set(k_core(n, edges, k + 1))
        if not b <= a:
            nested_ok = False
            break
    if not nested_ok:
        break
check("k-cores are nested: (k+1)-core is a subset of the k-core", nested_ok)

# --- every vertex in the k-core has >= k neighbours inside it --------------
rng = LCG(555)
degree_ok = True
for _ in range(200):
    n = rng.randint(1, 10)
    edges = random_graph(rng, n, 2, 4)
    adj = _adj(n, edges)
    deg = degeneracy(n, edges)
    for k in range(1, deg + 1):
        core = set(k_core(n, edges, k))
        for v in core:
            inside = sum(1 for w in adj[v] if w in core)
            if inside < k:
                degree_ok = False
                break
        if not degree_ok:
            break
    if not degree_ok:
        break
check("every k-core vertex has at least k neighbours within the core", degree_ok)

# --- degeneracy ordering property ------------------------------------------
# each vertex has at most `degeneracy` neighbours appearing later in the order
rng = LCG(31337)
order_ok = True
for _ in range(200):
    n = rng.randint(1, 12)
    edges = random_graph(rng, n, 2, 4)
    adj = _adj(n, edges)
    order = degeneracy_ordering(n, edges)
    pos = {v: i for i, v in enumerate(order)}
    deg = degeneracy(n, edges)
    if sorted(order) != list(range(n)):
        order_ok = False
        break
    for v in range(n):
        later = sum(1 for w in adj[v] if pos[w] > pos[v])
        if later > deg:
            order_ok = False
            break
    if not order_ok:
        break
check("degeneracy ordering: each vertex has <= degeneracy later neighbours", order_ok)

# --- shells partition all vertices -----------------------------------------
rng = LCG(99)
shell_ok = True
for _ in range(200):
    n = rng.randint(1, 10)
    edges = random_graph(rng, n, 2, 4)
    sh = shells(n, edges)
    covered = sorted(v for vs in sh.values() for v in vs)
    if covered != list(range(n)):
        shell_ok = False
        break
check("k-shells partition every vertex exactly once", shell_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all k_core tests passed")

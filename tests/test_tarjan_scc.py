"""Tests for tarjan_scc: SCCs vs brute reachability, condensation is a DAG, topological order."""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tarjan_scc import (strongly_connected_components, condensation, topological_sort,
                        has_cycle, is_dag)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 88
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def reachable(n, edges, s):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    seen = {s}
    q = deque([s])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    return seen


def brute_scc_partition(n, edges):
    # two vertices are in the same SCC iff each reaches the other
    reach = [reachable(n, edges, s) for s in range(n)]
    comp = [-1] * n
    cid = 0
    for i in range(n):
        if comp[i] != -1:
            continue
        comp[i] = cid
        for j in range(i + 1, n):
            if comp[j] == -1 and j in reach[i] and i in reach[j]:
                comp[j] = cid
        cid += 1
    return comp


def same_partition(comp_a, comp_b):
    # two labelings describe the same partition iff the induced equivalence is identical
    n = len(comp_a)
    for i in range(n):
        for j in range(n):
            if (comp_a[i] == comp_a[j]) != (comp_b[i] == comp_b[j]):
                return False
    return True


# --- known graph -----------------------------------------------------------
edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
comps = strongly_connected_components(6, edges)
comp_sets = sorted(sorted(c) for c in comps)
check("known graph SCCs are {0,1,2} and {3,4,5}", comp_sets == [[0, 1, 2], [3, 4, 5]])

# --- SCC partition matches brute-force mutual reachability -----------------
ok = True
for _ in range(100):
    n = 3 + int(rng() * 7)
    m = int(rng() * n * 2)
    edges = []
    for _ in range(m):
        u = int(rng() * n)
        v = int(rng() * n)
        edges.append((u, v))
    comps = strongly_connected_components(n, edges)
    comp_of = [0] * n
    for cid, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = cid
    brute = brute_scc_partition(n, edges)
    # every vertex must be covered exactly once
    covered = sorted(v for comp in comps for v in comp)
    if covered != list(range(n)):
        ok = False
        break
    if not same_partition(comp_of, brute):
        ok = False
        break
check("SCC partition matches brute-force mutual reachability over 100 random graphs", ok)

# --- condensation is always a DAG ------------------------------------------
ok = True
for _ in range(100):
    n = 3 + int(rng() * 7)
    edges = [(int(rng() * n), int(rng() * n)) for _ in range(int(rng() * n * 2))]
    _, dag, comps = condensation(n, edges)
    n_comp = len(comps)
    if has_cycle(n_comp, list(dag)):
        ok = False
        break
check("condensation is always acyclic (a DAG)", ok)

# --- SCCs returned in reverse topological order of the condensation --------
# for the known graph, component containing 0 depends on nothing downstream except {3,4,5}
edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
comps = strongly_connected_components(6, edges)
comp_of, dag, _ = condensation(6, edges)
# in reverse topological order, for every DAG edge (a,b) a must appear AFTER b in the list
order_index = {}
for pos, comp in enumerate(comps):
    order_index[comp_of[comp[0]]] = pos
rev_topo_ok = all(order_index[a] > order_index[b] for (a, b) in dag)
check("SCCs are in reverse topological order of the condensation", rev_topo_ok)

# --- a single big cycle is one component -----------------------------------
cycle = [(i, (i + 1) % 10) for i in range(10)]
comps = strongly_connected_components(10, cycle)
check("a 10-cycle is a single SCC", len(comps) == 1 and sorted(comps[0]) == list(range(10)))

# --- a DAG has all singleton SCCs ------------------------------------------
dag_edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
comps = strongly_connected_components(5, dag_edges)
check("a DAG has only singleton SCCs", all(len(c) == 1 for c in comps))
check("a DAG is reported acyclic", is_dag(5, dag_edges))

# --- topological sort validity ---------------------------------------------
order = topological_sort(5, dag_edges)
pos = {v: i for i, v in enumerate(order)}
check("topological order respects all edges", all(pos[u] < pos[v] for u, v in dag_edges))
check("topological order is a permutation", sorted(order) == list(range(5)))
check("cyclic graph has no topological order", topological_sort(2, [(0, 1), (1, 0)]) is None)

# --- disconnected vertices -------------------------------------------------
comps = strongly_connected_components(4, [(0, 1)])
check("isolated vertices are their own SCCs", len(comps) == 4)

# --- self-loop -------------------------------------------------------------
comps = strongly_connected_components(2, [(0, 0), (0, 1)])
check("self-loop vertex is its own SCC", sorted(sorted(c) for c in comps) == [[0], [1]])
check("self-loop counts as a cycle", has_cycle(1, [(0, 0)]))

# --- larger graph, no recursion limit --------------------------------------
big_cycle = [(i, (i + 1) % 5000) for i in range(5000)]
comps = strongly_connected_components(5000, big_cycle)
check("5000-cycle is one SCC (iterative DFS, no recursion limit)", len(comps) == 1)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all tarjan_scc tests passed")

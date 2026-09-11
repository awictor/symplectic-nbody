"""Tests for toposort.py -- topological sorting of a DAG.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Orders are validated
edge-by-edge; both algorithms are checked over hundreds of random DAGs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import toposort as T  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def build(edges, extra_nodes=()):
    g = T.DAG()
    for n in extra_nodes:
        g.add_node(n)
    for u, v in edges:
        g.add_edge(u, v)
    return g


# --- basic ordering ---------------------------------------------------------
g = build([("a", "b"), ("b", "c"), ("a", "c")])
k = T.kahn_sort(g)
d = T.dfs_sort(g)
check("Kahn returns all nodes", set(k) == {"a", "b", "c"})
check("DFS returns all nodes", set(d) == {"a", "b", "c"})
check("Kahn order is valid", T.is_valid_order(g, k))
check("DFS order is valid", T.is_valid_order(g, d))
check("a linear chain sorts in order", T.kahn_sort(build([("x", "y"), ("y", "z")])) == ["x", "y", "z"])
check("single node sorts to itself", T.kahn_sort(build([], extra_nodes=["solo"])) == ["solo"])
check("empty graph sorts to empty", T.kahn_sort(T.DAG()) == [])

# --- both algorithms agree on validity over many random DAGs ---------------
def random_dag(seed, n, prob):
    """A random DAG: only edges i->j with i<j, guaranteeing acyclicity."""
    g = T.DAG()
    for i in range(n):
        g.add_node(i)
    state = seed
    for i in range(n):
        for j in range(i + 1, n):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            if (state >> 16) % 100 < prob:
                g.add_edge(i, j)
    return g


kahn_ok = dfs_ok = True
for seed in range(1, 300):
    g = random_dag(seed, 10, 30)
    if not T.is_valid_order(g, T.kahn_sort(g)):
        kahn_ok = False
    if not T.is_valid_order(g, T.dfs_sort(g)):
        dfs_ok = False
check("Kahn produces a valid order on 299 random DAGs", kahn_ok)
check("DFS produces a valid order on 299 random DAGs", dfs_ok)

# --- is_valid_order rejects bad orders -------------------------------------
g = build([("a", "b"), ("b", "c")])
check("a reversed order is rejected", not T.is_valid_order(g, ["c", "b", "a"]))
check("an incomplete order is rejected", not T.is_valid_order(g, ["a", "b"]))
check("a correct order is accepted", T.is_valid_order(g, ["a", "b", "c"]))

# --- cycle detection --------------------------------------------------------
cyc = build([("a", "b"), ("b", "c"), ("c", "a")])
check("has_cycle detects a 3-cycle", T.has_cycle(cyc))
check("has_cycle is False for a DAG", not T.has_cycle(build([("a", "b"), ("a", "c")])))
check("a self-loop is a cycle", T.has_cycle(build([("a", "a")])))
try:
    T.kahn_sort(cyc)
    check("Kahn raises on a cycle", False)
except ValueError:
    check("Kahn raises on a cycle", True)
try:
    T.dfs_sort(cyc)
    check("DFS raises on a cycle", False)
except ValueError:
    check("DFS raises on a cycle", True)
# a cycle buried in a larger graph
big_cyc = build([("s", "a"), ("a", "b"), ("b", "c"), ("c", "a"), ("b", "t")])
check("detects a cycle inside a bigger graph", T.has_cycle(big_cyc))

# --- longest / critical path -----------------------------------------------
p = build([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")])
dur = {"A": 3, "B": 2, "C": 5, "D": 1, "E": 4}
length, path = T.longest_path(p, dur)
check("critical path is A->C->D->E", path == ["A", "C", "D", "E"])
check("critical path length is 13", length == 13)
# unit weights: longest path is the number of nodes on the longest chain
ulen, upath = T.longest_path(build([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]))
check("unit-weight longest path counts nodes on the longest chain", ulen == 4)
check("unit-weight longest path is the full chain", upath == ["a", "b", "c", "d"])
try:
    T.longest_path(cyc)
    check("longest_path raises on a cycle", False)
except ValueError:
    check("longest_path raises on a cycle", True)

# --- in-degrees -------------------------------------------------------------
g = build([("a", "c"), ("b", "c"), ("c", "d")])
deg = g.in_degrees()
check("in-degree counts incoming edges", deg["c"] == 2 and deg["d"] == 1 and deg["a"] == 0)

# --- a realistic dependency example ----------------------------------------
deps = build([
    ("boot", "kernel"), ("kernel", "drivers"), ("kernel", "filesystem"),
    ("drivers", "network"), ("filesystem", "network"), ("network", "services"),
])
order = T.kahn_sort(deps)
check("dependency order boots before kernel", order.index("boot") < order.index("kernel"))
check("dependency order is valid", T.is_valid_order(deps, order))
check("network waits for both drivers and filesystem",
      order.index("network") > order.index("drivers") and order.index("network") > order.index("filesystem"))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall toposort tests passed")

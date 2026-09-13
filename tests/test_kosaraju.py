"""Tests for Kosaraju SCC: matches Tarjan, components strongly connected + maximal, condensation is a DAG."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kosaraju import (  # noqa: E402
    strongly_connected_components,
    component_index,
    condensation,
    is_strongly_connected_set,
)
from tarjan_scc import strongly_connected_components as tarjan_scc, is_dag  # noqa: E402


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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _partition(comps):
    """Canonical form: a frozenset of frozensets, so ordering does not matter."""
    return frozenset(frozenset(c) for c in comps)


def main():
    # ---- 1. matches Tarjan's partition on random directed graphs ------------------------
    rng = _lcg(1)
    ok = True
    for trial in range(40):
        n = 3 + int(rng() * 12)
        m = int(rng() * n * 2)
        edges = []
        for _ in range(m):
            u = int(rng() * n)
            v = int(rng() * n)
            edges.append((u, v))
        k = _partition(strongly_connected_components(n, edges))
        t = _partition(tarjan_scc(n, edges))
        if k != t:
            ok = False
            check("Kosaraju == Tarjan partition", False, f"n={n} edges={edges}")
            break
    if ok:
        check("Kosaraju == Tarjan partition (40 random graphs)", True)

    # ---- 2. every component is strongly connected ---------------------------------------
    rng = _lcg(2)
    n = 10
    edges = [(int(rng() * n), int(rng() * n)) for _ in range(20)]
    comps = strongly_connected_components(n, edges)
    check("all components strongly connected",
          all(is_strongly_connected_set(n, edges, c) for c in comps))

    # ---- 3. components partition the vertices exactly -----------------------------------
    allv = sorted(v for c in comps for v in c)
    check("components partition all vertices", allv == list(range(n)),
          f"{allv}")
    # disjoint
    seen = set()
    disjoint = True
    for c in comps:
        for v in c:
            if v in seen:
                disjoint = False
            seen.add(v)
    check("components are disjoint", disjoint)

    # ---- 4. single big cycle is one component -------------------------------------------
    n = 6
    cycle = [(i, (i + 1) % n) for i in range(n)]
    comps = strongly_connected_components(n, cycle)
    check("cycle -> single SCC", len(comps) == 1 and len(comps[0]) == n, f"{comps}")

    # ---- 5. a DAG has all singleton components ------------------------------------------
    dag = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    comps = strongly_connected_components(5, dag)
    check("DAG -> all singletons", all(len(c) == 1 for c in comps), f"{comps}")

    # ---- 6. classic example with known SCCs ---------------------------------------------
    # 0->1->2->0 (SCC {0,1,2}), 3->4->3 (SCC {3,4}), 2->3 bridge
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)]
    comps = strongly_connected_components(5, edges)
    part = _partition(comps)
    check("known SCCs {0,1,2},{3,4}",
          part == frozenset([frozenset([0, 1, 2]), frozenset([3, 4])]), f"{comps}")

    # ---- 7. components in topological order of the condensation -------------------------
    # {0,1,2} reaches {3,4}, so it must come first
    idx = component_index(5, edges)
    check("source SCC has smaller index", idx[0] < idx[3], f"idx={idx}")

    # ---- 8. condensation is a DAG -------------------------------------------------------
    rng = _lcg(9)
    n = 12
    edges = [(int(rng() * n), int(rng() * n)) for _ in range(24)]
    num, cedges = condensation(n, edges)
    check("condensation is acyclic", is_dag(num, cedges), f"{cedges}")
    # condensation edges only go from lower to higher index (topological)
    check("condensation edges go source->sink", all(a < b for a, b in cedges),
          f"{cedges}")

    # ---- 9. self-loops and multi-edges handled ------------------------------------------
    edges = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    comps = strongly_connected_components(3, edges)
    check("self-loops handled",
          _partition(comps) == frozenset([frozenset([0, 1]), frozenset([2])]), f"{comps}")

    # ---- 10. empty graph and single vertex ----------------------------------------------
    check("single vertex -> one component", strongly_connected_components(1, []) == [[0]])
    check("no edges -> all singletons",
          _partition(strongly_connected_components(4, [])) ==
          frozenset([frozenset([0]), frozenset([1]), frozenset([2]), frozenset([3])]))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

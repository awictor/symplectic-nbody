"""Tests for eulerian: existence conditions + Hierholzer trails vs brute validation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eulerian import (undirected_euler_status, undirected_eulerian_trail,
                      directed_euler_status, directed_eulerian_trail,
                      is_valid_undirected_trail, is_valid_directed_trail)

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


def brute_undirected_status(n, edges):
    """Existence by definition: connected (non-isolated) + count odd-degree vertices."""
    from collections import defaultdict
    adj = defaultdict(list)
    noniso = set()
    deg = defaultdict(int)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
        noniso.add(u)
        noniso.add(v)
    if noniso:
        start = next(iter(noniso))
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for w in adj[x]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        if not (noniso <= seen):
            return "none"
    odd = sum(1 for v in deg if deg[v] % 2 == 1)
    if odd == 0:
        return "circuit"
    if odd == 2:
        return "path"
    return "none"


# --- known cases ------------------------------------------------------------
check("triangle has an Eulerian circuit", undirected_euler_status(3, [(0, 1), (1, 2), (2, 0)]) == "circuit")
check("path graph has an Eulerian path", undirected_euler_status(3, [(0, 1), (1, 2)]) == "path")
check("K4 (all degree 3) has neither",
      undirected_euler_status(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]) == "none")
# The Seven Bridges of Konigsberg: 4 land masses, all odd degree -> no trail (Euler 1736)
konigsberg = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
check("Seven Bridges of Konigsberg has no Eulerian trail", undirected_euler_status(4, konigsberg) == "none")

check("directed cycle has an Eulerian circuit",
      directed_euler_status(3, [(0, 1), (1, 2), (2, 0)]) == "circuit")
check("directed path has an Eulerian path", directed_euler_status(3, [(0, 1), (1, 2)]) == "path")
check("unbalanced directed graph has neither",
      directed_euler_status(3, [(0, 1), (0, 2)]) == "none")

# --- undirected: status matches brute, trail is valid ----------------------
rng = LCG(2026)
und_status_ok = und_trail_ok = True
saw_c = saw_p = saw_n = False
for _ in range(500):
    n = rng.randint(1, 7)
    m = rng.randint(0, 10)
    edges = []
    for _ in range(m):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.append((u, v))
    st = undirected_euler_status(n, edges)
    if st != brute_undirected_status(n, edges):
        und_status_ok = False
        print(f"  status mismatch: {st} vs brute for edges={edges}")
        break
    trail = undirected_eulerian_trail(n, edges)
    if st == "none":
        saw_n = True
        if trail is not None and edges:
            und_trail_ok = False
            break
    else:
        if st == "circuit":
            saw_c = True
        else:
            saw_p = True
        if not is_valid_undirected_trail(edges, trail):
            und_trail_ok = False
            print(f"  invalid undirected trail for edges={edges}: {trail}")
            break
check("undirected Euler status matches the definition (500 random graphs)", und_status_ok)
check("undirected trails use every edge exactly once (valid)", und_trail_ok)
check("undirected suite saw circuit, path, and none", saw_c and saw_p and saw_n)

# circuits start and end at the same vertex; paths at the two odd vertices
edges = [(0, 1), (1, 2), (2, 0)]
tr = undirected_eulerian_trail(3, edges)
check("undirected circuit returns to its start", tr[0] == tr[-1])
edges = [(0, 1), (1, 2)]
tr = undirected_eulerian_trail(3, edges)
check("undirected path ends at the two odd-degree vertices", {tr[0], tr[-1]} == {0, 2})

# --- directed: trail validity ----------------------------------------------
rng = LCG(4242)
dir_trail_ok = True
saw_dc = saw_dp = False
for _ in range(500):
    n = rng.randint(1, 7)
    m = rng.randint(0, 10)
    edges = []
    for _ in range(m):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.append((u, v))
    st = directed_euler_status(n, edges)
    trail = directed_eulerian_trail(n, edges)
    if st == "none":
        if trail is not None and edges:
            dir_trail_ok = False
            break
    else:
        if st == "circuit":
            saw_dc = True
        else:
            saw_dp = True
        if not is_valid_directed_trail(edges, trail):
            dir_trail_ok = False
            print(f"  invalid directed trail for edges={edges}: {trail}")
            break
check("directed trails follow real edges and use each exactly once (500 graphs)", dir_trail_ok)
check("directed suite saw both circuit and path", saw_dc and saw_dp)

# --- multigraph handling ----------------------------------------------------
# doubled edge between two vertices: each vertex has even degree -> circuit
multi = [(0, 1), (0, 1)]
check("undirected multigraph (doubled edge) is a circuit", undirected_euler_status(2, multi) == "circuit")
tr = undirected_eulerian_trail(2, multi)
check("undirected multigraph trail is valid", is_valid_undirected_trail(multi, tr))

# --- empty and trivial ------------------------------------------------------
check("empty undirected graph is a (trivial) circuit", undirected_euler_status(1, []) == "circuit")
check("empty directed graph is a (trivial) circuit", directed_euler_status(1, []) == "circuit")

# --- a bigger Eulerian circuit solves and is valid -------------------------
# two triangles sharing a vertex: all even degree -> circuit
edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
check("two triangles sharing a vertex form a circuit", undirected_euler_status(5, edges) == "circuit")
tr = undirected_eulerian_trail(5, edges)
check("shared-vertex circuit trail is valid and closed",
      is_valid_undirected_trail(edges, tr) and tr[0] == tr[-1])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all eulerian tests passed")

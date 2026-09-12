"""Tests for yen_ksp: K shortest loopless paths vs brute enumeration of all simple paths."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from yen_ksp import k_shortest_paths, brute_k_shortest

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


def costs_of(paths):
    return [c for c, _ in paths]


def loopless(path):
    return len(path) == len(set(path))


def valid_path(edges, path):
    adj = {}
    for u, v, w in edges:
        adj.setdefault((u, v), w)
    return all((path[i], path[i + 1]) in adj for i in range(len(path) - 1))


# --- known case -------------------------------------------------------------
edges = [(0, 1, 3), (0, 2, 2), (1, 2, 1), (1, 3, 4), (2, 3, 2), (2, 1, 1), (3, 4, 2), (1, 4, 7)]
ksp = k_shortest_paths(5, edges, 0, 4, 3)
check("known graph: 3 shortest path costs are [6, 8, 9]", costs_of(ksp) == [6, 8, 9])
check("known graph: shortest path is 0-2-3-4", ksp[0][1] == [0, 2, 3, 4])

# --- source == target -------------------------------------------------------
check("source equals target: single trivial path of cost 0",
      k_shortest_paths(3, [(0, 1, 1)], 0, 0, 3) == [(0, [0])])

# --- unreachable target -----------------------------------------------------
check("unreachable target: no paths", k_shortest_paths(3, [(0, 1, 1)], 0, 2, 3) == [])

# --- fewer than K paths exist ----------------------------------------------
# only one path 0->1->2
res = k_shortest_paths(3, [(0, 1, 1), (1, 2, 1)], 0, 2, 5)
check("only one path exists though K=5", len(res) == 1 and res[0] == (2, [0, 1, 2]))

# --- exhaustive validation vs brute ----------------------------------------
rng = LCG(2026)
match_ok = loopless_ok = valid_ok = sorted_ok = True
saw_multi = False
for _ in range(400):
    n = rng.randint(2, 7)
    m = rng.randint(1, 12)
    edges = []
    seen_uv = set()
    for _ in range(m):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and (u, v) not in seen_uv:      # simple graph: no parallel edges
            seen_uv.add((u, v))
            edges.append((u, v, rng.randint(1, 9)))
    source = 0
    target = n - 1
    K = rng.randint(1, 5)
    yen = k_shortest_paths(n, edges, source, target, K)
    brute = brute_k_shortest(n, edges, source, target, K)
    # compare cost sequences (paths may tie, but the multiset of costs must match)
    if costs_of(yen) != costs_of(brute):
        match_ok = False
        print(f"  cost mismatch: yen={costs_of(yen)} brute={costs_of(brute)} edges={edges} K={K}")
        break
    for c, p in yen:
        if not loopless(p):
            loopless_ok = False
        if not valid_path(edges, p):
            valid_ok = False
        if p[0] != source or p[-1] != target:
            valid_ok = False
    if costs_of(yen) != sorted(costs_of(yen)):
        sorted_ok = False
    if len(yen) >= 2:
        saw_multi = True
    if not (match_ok and loopless_ok and valid_ok and sorted_ok):
        break
check("Yen's K costs match brute-force enumeration (400 random graphs)", match_ok)
check("all returned paths are loopless (simple)", loopless_ok)
check("all returned paths are valid source-to-target walks", valid_ok)
check("returned paths are in non-decreasing cost order", sorted_ok)
check("suite produced graphs with multiple alternative paths", saw_multi)

# --- paths are distinct -----------------------------------------------------
rng = LCG(4242)
distinct_ok = True
for _ in range(200):
    n = rng.randint(2, 7)
    edges = []
    for _ in range(rng.randint(1, 12)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.append((u, v, rng.randint(1, 9)))
    yen = k_shortest_paths(n, edges, 0, n - 1, 5)
    paths = [tuple(p) for _, p in yen]
    if len(paths) != len(set(paths)):
        distinct_ok = False
        break
check("returned K paths are all distinct", distinct_ok)

# --- the first path equals a plain shortest path ---------------------------
rng = LCG(777)
first_ok = True
for _ in range(200):
    n = rng.randint(2, 7)
    edges = []
    for _ in range(rng.randint(1, 12)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.append((u, v, rng.randint(1, 9)))
    yen = k_shortest_paths(n, edges, 0, n - 1, 3)
    brute = brute_k_shortest(n, edges, 0, n - 1, 1)
    if brute:
        if not yen or yen[0][0] != brute[0][0]:
            first_ok = False
            break
    else:
        if yen:
            first_ok = False
            break
check("the first Yen path is a true shortest path", first_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all yen_ksp tests passed")

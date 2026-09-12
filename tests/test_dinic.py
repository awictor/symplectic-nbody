"""Tests for dinic: max flow vs Edmonds-Karp, max-flow/min-cut theorem, bipartite matching."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dinic import (max_flow, min_cut_value, bipartite_matching_size, edmonds_karp, Dinic)

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


def cut_capacity(n, edges, source_side):
    """Total capacity of edges crossing from the source side to the other side (directed)."""
    ss = set(source_side)
    return sum(c for (u, v, c) in edges if u in ss and v not in ss)


# --- known case -------------------------------------------------------------
edges = [(0, 1, 3), (0, 2, 2), (1, 2, 1), (1, 3, 2), (2, 3, 3), (3, 4, 4), (1, 4, 1)]
check("known network: max flow 5", max_flow(5, edges, 0, 4) == 5)
check("Dinic equals Edmonds-Karp on the known network",
      max_flow(5, edges, 0, 4) == edmonds_karp(5, edges, 0, 4))

check("source == sink: flow 0", max_flow(3, [(0, 1, 5)], 1, 1) == 0)
check("disconnected sink: flow 0", max_flow(4, [(0, 1, 5)], 0, 3) == 0)

# classic CLRS network: max flow 23
clrs = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12),
        (3, 2, 9), (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
check("CLRS textbook network: max flow 23", max_flow(6, clrs, 0, 5) == 23)

# --- exhaustive vs Edmonds-Karp --------------------------------------------
rng = LCG(2026)
flow_ok = True
for _ in range(400):
    n = rng.randint(2, 8)
    edges = []
    for _ in range(rng.randint(1, 16)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.append((u, v, rng.randint(1, 12)))
    src, snk = 0, n - 1
    if max_flow(n, edges, src, snk) != edmonds_karp(n, edges, src, snk):
        flow_ok = False
        print(f"  mismatch: dinic={max_flow(n, edges, src, snk)} "
              f"ek={edmonds_karp(n, edges, src, snk)} edges={edges}")
        break
check("Dinic max flow equals Edmonds-Karp (400 random networks)", flow_ok)

# --- max-flow / min-cut theorem --------------------------------------------
rng = LCG(4242)
mincut_ok = True
for _ in range(300):
    n = rng.randint(2, 8)
    edges = []
    for _ in range(rng.randint(1, 16)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.append((u, v, rng.randint(1, 10)))
    src, snk = 0, n - 1
    flow, side = min_cut_value(n, edges, src, snk)
    # the source-side must contain the source and not the sink (when flow reaches the sink)
    if src not in side:
        mincut_ok = False
        break
    # the capacity of the returned cut must equal the flow (max-flow min-cut theorem)
    if flow > 0 and cut_capacity(n, edges, side) != flow:
        mincut_ok = False
        print(f"  cut capacity {cut_capacity(n, edges, side)} != flow {flow} edges={edges}")
        break
    if flow > 0 and snk in side:
        mincut_ok = False
        break
check("max-flow equals the capacity of the returned min cut (300 networks)", mincut_ok)

# --- bipartite matching via max flow ---------------------------------------
# a perfect matching exists here
pairs = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)]
check("bipartite matching: perfect matching of size 3", bipartite_matching_size(3, 3, pairs) == 3)

# a bottleneck: two lefts only reach one right
check("bipartite matching bottleneck: size 1", bipartite_matching_size(2, 2, [(0, 0), (1, 0)]) == 1)

# match Dinic bipartite matching against a brute maximum matching
def brute_matching(n_left, n_right, pairs):
    from itertools import combinations
    adj = {}
    for li, rj in pairs:
        adj.setdefault(li, set()).add(rj)
    best = 0
    # try to match as many left vertices as possible (Hungarian-style augmenting brute)
    match_r = {}

    def try_aug(u, seen):
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                if v not in match_r or try_aug(match_r[v], seen):
                    match_r[v] = u
                    return True
        return False

    for u in range(n_left):
        try_aug(u, set())
    return len(match_r)


rng = LCG(777)
match_ok = True
for _ in range(200):
    nl = rng.randint(1, 6)
    nr = rng.randint(1, 6)
    pairs = []
    for li in range(nl):
        for rj in range(nr):
            if rng.rand() % 2 == 0:
                pairs.append((li, rj))
    if bipartite_matching_size(nl, nr, pairs) != brute_matching(nl, nr, pairs):
        match_ok = False
        break
check("bipartite matching size matches an augmenting-path reference (200 graphs)", match_ok)

# --- parallel edges combine -------------------------------------------------
d = Dinic(2)
d.add_edge(0, 1, 3)
d.add_edge(0, 1, 5)
check("parallel edges add capacity", d.max_flow(0, 1) == 8)

# --- larger network solves fast --------------------------------------------
rng = LCG(31337)
n = 200
edges = []
for _ in range(1500):
    u = rng.randint(0, n - 1)
    v = rng.randint(0, n - 1)
    if u != v:
        edges.append((u, v, rng.randint(1, 20)))
check("200-node network: Dinic and Edmonds-Karp agree",
      max_flow(n, edges, 0, n - 1) == edmonds_karp(n, edges, 0, n - 1))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all dinic tests passed")

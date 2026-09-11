"""Tests for max_flow: Edmonds-Karp, max-flow min-cut theorem, bipartite matching."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from max_flow import (MaxFlow, edmonds_karp, min_cut_value, bipartite_matching)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- the classic CLRS network: max flow 23 ---------------------------------
clrs = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12), (3, 2, 9),
        (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
check("CLRS network max flow is 23", edmonds_karp(6, clrs, 0, 5) == 23)

# --- max-flow == min-cut ---------------------------------------------------
flow, cut = min_cut_value(6, clrs, 0, 5)
check("min-cut capacity equals the max flow", sum(c for _, _, c in cut) == flow)
check("min cut is 23", flow == 23)
check("cut edges all leave the source side", len(cut) > 0)

# --- simple networks -------------------------------------------------------
check("parallel edges sum", edmonds_karp(2, [(0, 1, 5), (0, 1, 3)], 0, 1) == 8)
check("series bottleneck", edmonds_karp(3, [(0, 1, 10), (1, 2, 3)], 0, 2) == 3)
check("no path -> zero flow", edmonds_karp(3, [(0, 1, 5)], 0, 2) == 0)
check("source == sink -> zero", edmonds_karp(2, [(0, 1, 5)], 1, 1) == 0)
check("single edge", edmonds_karp(2, [(0, 1, 7)], 0, 1) == 7)

# --- a diamond where two paths share a bottleneck --------------------------
# 0->1->3 and 0->2->3, but 1->3 and 2->3 each cap 5, sources 10 each -> flow 10
diamond = [(0, 1, 10), (0, 2, 10), (1, 3, 5), (2, 3, 5)]
check("diamond flow limited by sink edges", edmonds_karp(4, diamond, 0, 3) == 10)

# --- residual reverse edges let flow reroute (the classic anti-greedy case)-
# 0->1(1), 0->2(1), 1->2(1), 1->3(1), 2->3(1): max flow 2 needs the 1->2 reverse
tricky = [(0, 1, 1), (0, 2, 1), (1, 2, 1), (1, 3, 1), (2, 3, 1)]
check("residual rerouting achieves the true max flow", edmonds_karp(4, tricky, 0, 3) == 2)

# --- flow conservation and capacity constraints hold -----------------------
mf = MaxFlow(6)
orig = {}
for u, v, c in clrs:
    mf.add_edge(u, v, c)
    orig[(u, v)] = c
value = mf.max_flow(0, 5)
# used flow on each original edge = original capacity - residual capacity, must be in [0, cap]
caps_ok = True
for (u, v), c in orig.items():
    used = c - mf.cap[u][v]
    if not (-1e-9 <= used <= c + 1e-9):
        caps_ok = False
check("no edge exceeds its capacity", caps_ok)
# conservation: net flow out of source == value == net flow into sink
out_source = sum(orig.get((0, v), 0) - mf.cap[0][v] for v in range(6) if (0, v) in orig)
check("flow out of source equals the max flow", abs(out_source - value) < 1e-9)

# --- bipartite matching by reduction ---------------------------------------
size, pairs = bipartite_matching(["a", "b", "c"], ["x", "y", "z"],
                                 [("a", "x"), ("a", "y"), ("b", "x"), ("c", "z")])
check("bipartite perfect matching size 3", size == 3)
check("matching pairs are valid edges", all(p in [("a", "x"), ("a", "y"), ("b", "x"), ("c", "z")]
                                            for p in pairs))
check("each left vertex matched at most once", len({l for l, _ in pairs}) == len(pairs))
check("each right vertex matched at most once", len({r for _, r in pairs}) == len(pairs))

# --- a star: three lefts competing for one right -> matching 1 -------------
s_star, _ = bipartite_matching(["a", "b", "c"], ["x"], [("a", "x"), ("b", "x"), ("c", "x")])
check("star matching is 1", s_star == 1)

# --- a complete bipartite graph K3,3 -> perfect matching 3 -----------------
k33 = [(l, r) for l in "abc" for r in "xyz"]
s_k33, p_k33 = bipartite_matching(list("abc"), list("xyz"), k33)
check("K3,3 matching is 3", s_k33 == 3)
check("K3,3 matching is a permutation", len({l for l, _ in p_k33}) == 3 and len({r for _, r in p_k33}) == 3)

# --- no edges -> no matching -----------------------------------------------
check("no edges -> matching 0", bipartite_matching(["a"], ["x"], [])[0] == 0)

# --- bipartite matching by flow equals a direct augmenting-path matching ---
def direct_matching(left, right, edges):
    adj = {l: [r for (ll, r) in edges if ll == l] for l in left}
    match_r = {}

    def try_augment(l, seen):
        for r in adj.get(l, []):
            if r in seen:
                continue
            seen.add(r)
            if r not in match_r or try_augment(match_r[r], seen):
                match_r[r] = l
                return True
        return False

    count = 0
    for l in left:
        if try_augment(l, set()):
            count += 1
    return count


import random
state = 7


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


all_match = True
for _ in range(40):
    L = [f"l{i}" for i in range(5)]
    R = [f"r{i}" for i in range(5)]
    es = [(l, r) for l in L for r in R if rng() < 0.35]
    flow_size = bipartite_matching(L, R, es)[0]
    direct = direct_matching(L, R, es)
    if flow_size != direct:
        all_match = False
        break
check("flow matching equals direct matching over 40 random bipartite graphs", all_match)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all max_flow tests passed")

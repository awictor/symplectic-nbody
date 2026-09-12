"""Tests for min_cost_flow: SPFA successive-shortest-paths vs Edmonds-Karp, brute force, Hungarian."""

import os
import sys
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from min_cost_flow import MinCostFlow, assignment_min_cost
from max_flow import edmonds_karp
from hungarian import min_cost as hungarian_min_cost

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


# --- known small case -------------------------------------------------------
m = MinCostFlow(4)
for u, v, cap, cost in [(0, 1, 3, 1), (0, 2, 2, 2), (1, 3, 2, 1), (2, 3, 3, 1), (1, 2, 1, 1)]:
    m.add_edge(u, v, cap, cost)
f, c = m.min_cost_max_flow(0, 3)
check("known network: flow 5", f == 5)
# cheapest routing: 2 units 0-1-3 (cost 2*2=4), 1 unit 0-1-2-3 (cost 3), 2 units 0-2-3 (cost 2*3=6)
check("known network: cost 13", c == 13)

# --- flow value equals Edmonds-Karp max-flow -------------------------------
rng = LCG(2026)
flow_ok = True
for _ in range(200):
    n = rng.randint(2, 7)
    ek_edges = []
    m = MinCostFlow(n)
    n_edges = rng.randint(1, 12)
    for _ in range(n_edges):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        cap = rng.randint(1, 6)
        cost = rng.randint(0, 9)
        m.add_edge(u, v, cap, cost)
        ek_edges.append((u, v, cap))
    src, snk = 0, n - 1
    flow, _ = m.min_cost_max_flow(src, snk)
    if flow != edmonds_karp(n, ek_edges, src, snk):
        flow_ok = False
        break
check("min-cost-max-flow value equals Edmonds-Karp max-flow (200 random nets)", flow_ok)


# --- cost is minimal: brute force over all integer flows on tiny nets ------
def brute_min_cost_max_flow(n, edges, source, sink):
    """Exhaustive: find the max flow value, then the min cost achieving it, by enumerating integer
    flow assignments on each edge up to capacity. Only for tiny networks."""
    E = len(edges)

    best = {"flow": -1, "cost": None}

    def conserves(assign):
        # net flow at each node: 0 except +F at source, -F at sink
        net = [0] * n
        for (u, v, cap, cost), fl in zip(edges, assign):
            net[u] -= fl
            net[v] += fl
        F = net[sink]
        if net[source] != -F:
            return None
        for w in range(n):
            if w != source and w != sink and net[w] != 0:
                return None
        return F

    def rec(idx, assign):
        if idx == E:
            F = conserves(assign)
            if F is None or F < 0:
                return
            cost = sum(fl * e[3] for e, fl in zip(edges, assign))
            if F > best["flow"] or (F == best["flow"] and cost < best["cost"]):
                best["flow"] = F
                best["cost"] = cost
            return
        cap = edges[idx][2]
        for fl in range(cap + 1):
            assign.append(fl)
            rec(idx + 1, assign)
            assign.pop()

    rec(0, [])
    return best["flow"], best["cost"]


rng = LCG(4242)
brute_ok = True
for _ in range(120):
    n = rng.randint(2, 4)
    edges = []
    m = MinCostFlow(n)
    for _ in range(rng.randint(1, 5)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        cap = rng.randint(1, 3)
        cost = rng.randint(0, 5)
        m.add_edge(u, v, cap, cost)
        edges.append((u, v, cap, cost))
    src, snk = 0, n - 1
    flow, cost = m.min_cost_max_flow(src, snk)
    bf, bc = brute_min_cost_max_flow(n, edges, src, snk)
    if flow != bf or (bf > 0 and cost != bc):
        brute_ok = False
        print(f"  mismatch: fast=({flow},{cost}) brute=({bf},{bc}) edges={edges}")
        break
check("cost is provably minimal vs brute-force over all integer flows (120 tiny nets)", brute_ok)


# --- as assignment: matches the Hungarian algorithm -----------------------
rng = LCG(31415)
assign_ok = True
for _ in range(150):
    n = rng.randint(1, 5)
    cost = [[rng.randint(0, 20) for _ in range(n)] for _ in range(n)]
    total, assignment = assignment_min_cost(cost)
    if total != int(hungarian_min_cost(cost)):
        assign_ok = False
        break
    # assignment must be a permutation and its summed cost must equal `total`
    if sorted(assignment) != list(range(n)):
        assign_ok = False
        break
    if sum(cost[i][assignment[i]] for i in range(n)) != total:
        assign_ok = False
        break
check("bipartite assignment cost matches the Hungarian algorithm (150 matrices)", assign_ok)

# also cross-check the Hungarian result against brute permutation search on tiny matrices
rng = LCG(88)
perm_ok = True
for _ in range(60):
    n = rng.randint(1, 5)
    cost = [[rng.randint(0, 15) for _ in range(n)] for _ in range(n)]
    total, _ = assignment_min_cost(cost)
    best = min(sum(cost[i][p[i]] for i in range(n)) for p in permutations(range(n)))
    if total != best:
        perm_ok = False
        break
check("assignment cost equals the best over all permutations (brute, 60 matrices)", perm_ok)

# --- min-cost for a fixed target flow --------------------------------------
m = MinCostFlow(4)
for u, v, cap, cost in [(0, 1, 3, 1), (0, 2, 2, 2), (1, 3, 2, 1), (2, 3, 3, 1), (1, 2, 1, 1)]:
    m.add_edge(u, v, cap, cost)
# sending only 2 units should take the two cheapest unit-paths: 0-1-3 twice, cost 2*2=4
sent, cost = m.min_cost_for_flow(0, 3, 2)
check("min_cost_for_flow sends the target and picks cheapest paths", sent == 2 and cost == 4)

# requesting more than the network can carry returns the true max
m = MinCostFlow(3)
m.add_edge(0, 1, 2, 1)
m.add_edge(1, 2, 5, 1)
sent, _ = m.min_cost_for_flow(0, 2, 100)
check("min_cost_for_flow caps at the network capacity", sent == 2)

# --- zero-cost network reduces to plain max-flow ---------------------------
m = MinCostFlow(4)
for u, v, cap in [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3), (1, 2, 1)]:
    m.add_edge(u, v, cap, 0)
f, c = m.min_cost_max_flow(0, 3)
check("all-zero-cost network: cost 0, flow equals max-flow", c == 0 and f == 5)

# --- disconnected sink: no flow --------------------------------------------
m = MinCostFlow(4)
m.add_edge(0, 1, 5, 3)
f, c = m.min_cost_max_flow(0, 3)
check("unreachable sink: zero flow, zero cost", f == 0 and c == 0)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all min_cost_flow tests passed")

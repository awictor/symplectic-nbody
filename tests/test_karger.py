"""Tests for karger: randomized min cut, checked statistically against exact Stoer-Wagner."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from karger import karger_min_cut, karger_stein, cut_weight, _LCG
from stoer_wagner import min_cut as sw_min_cut

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def rand_connected(rng, n):
    """A random connected weighted graph (spanning path + extra edges)."""
    edges = []
    seen = set()
    for i in range(1, n):
        j = rng.randint(0, i - 1)
        edges.append((j, i, rng.randint(1, 6)))
        seen.add((min(i, j), max(i, j)))
    for _ in range(rng.randint(0, n)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and (min(u, v), max(u, v)) not in seen:
            seen.add((min(u, v), max(u, v)))
            edges.append((u, v, rng.randint(1, 6)))
    return edges


# --- known cases ------------------------------------------------------------
# two triangles joined by one edge: min cut is that single edge
bridge = [(0, 1, 1), (1, 2, 1), (2, 0, 1), (3, 4, 1), (4, 5, 1), (5, 3, 1), (2, 3, 1)]
cut, part = karger_min_cut(6, bridge)
check("two triangles + bridge: min cut is 1", cut == 1)
check("returned partition realises the reported cut", cut_weight(6, bridge, part[0]) == cut)
check("Karger-Stein also finds min cut 1", karger_stein(6, bridge)[0] == 1)

# a single edge: cut is its weight
check("single edge graph: cut equals its weight", karger_min_cut(2, [(0, 1, 7)])[0] == 7)

# weighted bottleneck
w = [(0, 1, 5), (1, 2, 5), (0, 2, 5), (2, 3, 2), (3, 4, 5), (4, 5, 5), (3, 5, 5)]
check("weighted graph: min cut is the light bridge (2)", karger_min_cut(6, w)[0] == 2)

# --- a Karger cut is ALWAYS a valid cut (>= true minimum) ------------------
rng = _LCG(2026)
never_below_ok = True
for _ in range(200):
    n = rng.randint(2, 9)
    edges = rand_connected(rng, n)
    exact = sw_min_cut(n, edges)[0]
    kc, part = karger_min_cut(n, edges, trials=n * n)
    # a contraction cut can never be below the true minimum, and must match its partition weight
    if kc < exact or cut_weight(n, edges, part[0]) != kc:
        never_below_ok = False
        print(f"  invalid: karger={kc} exact={exact} n={n} edges={edges}")
        break
check("a Karger cut is always valid and never below the true minimum (200 graphs)", never_below_ok)

# --- with enough trials, Karger EQUALS the exact minimum -------------------
rng = _LCG(4242)
exact_ok = True
tested = 0
for _ in range(120):
    n = rng.randint(2, 8)
    edges = rand_connected(rng, n)
    exact = sw_min_cut(n, edges)[0]
    # generous trial count so the probability of missing the min cut is tiny
    kc, _ = karger_min_cut(n, edges, trials=max(30, 3 * n * n))
    if kc != exact:
        exact_ok = False
        print(f"  did not reach exact: karger={kc} exact={exact} n={n} edges={edges}")
        break
    tested += 1
check(f"Karger reaches the exact Stoer-Wagner minimum given enough trials ({tested} graphs)",
      exact_ok)

# --- Karger-Stein matches the exact minimum too ----------------------------
# Karger-Stein is Monte Carlo (each call succeeds with probability ~1/log n), so it is used by
# repeating a few times and keeping the best -- exactly as done here.
rng = _LCG(777)
ks_ok = True
for _ in range(80):
    n = rng.randint(2, 8)
    edges = rand_connected(rng, n)
    exact = sw_min_cut(n, edges)[0]
    best = float("inf")
    best_part = None
    for rep in range(12):
        ks, part = karger_stein(n, edges, seed=rng.randint(1, 10 ** 9))
        if ks < best:
            best = ks
            best_part = part
    if best != exact:
        ks_ok = False
        print(f"  karger-stein missed: ks={best} exact={exact} n={n} edges={edges}")
        break
    # a Karger-Stein cut is always valid, never below the true minimum
    if best < exact or cut_weight(n, edges, best_part[0]) != best:
        ks_ok = False
        break
check("Karger-Stein reaches the exact minimum over a few repeats, with valid partition (80 graphs)",
      ks_ok)

# --- reproducibility --------------------------------------------------------
edges = rand_connected(_LCG(1), 7)
a = karger_min_cut(7, edges, trials=50, seed=999)[0]
b = karger_min_cut(7, edges, trials=50, seed=999)[0]
check("same seed gives the same result (reproducible)", a == b)

# --- partition is non-trivial (both sides non-empty) -----------------------
rng = _LCG(555)
part_ok = True
for _ in range(100):
    n = rng.randint(2, 8)
    edges = rand_connected(rng, n)
    _, (sa, sb) = karger_min_cut(n, edges, trials=n * n)
    if len(sa) == 0 or len(sb) == 0 or len(sa) + len(sb) != n:
        part_ok = False
        break
check("the cut partition splits all vertices into two non-empty sides", part_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all karger tests passed")

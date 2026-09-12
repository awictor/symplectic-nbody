"""Tests for weighted_dsu: difference-constraint solving vs brute BFS-offset reference."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weighted_dsu import WeightedDSU, ParityDSU, brute_process

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


# --- known case -------------------------------------------------------------
d = WeightedDSU(6)
check("union(0,1,5) accepted", d.union(0, 1, 5))
check("union(1,2,3) accepted", d.union(1, 2, 3))
check("diff(0,2) == 8 (5+3)", d.diff(0, 2) == 8)
check("consistent restatement union(0,2,8) accepted", d.union(0, 2, 8))
check("contradictory union(0,2,7) rejected", not d.union(0, 2, 7))
check("diff to an unconnected node is None", d.diff(0, 3) is None)
check("connected(0,2) true, connected(0,3) false", d.connected(0, 2) and not d.connected(0, 3))

# --- accept/reject decisions match brute -----------------------------------
rng = LCG(2026)
decisions_ok = True
for _ in range(500):
    n = rng.randint(2, 8)
    m = rng.randint(1, 20)
    cons = []
    for _ in range(m):
        x = rng.randint(0, n - 1)
        y = rng.randint(0, n - 1)
        dd = rng.randint(-10, 10)
        cons.append((x, y, dd))
    dsu = WeightedDSU(n)
    mine = [dsu.union(x, y, dv) for (x, y, dv) in cons]
    brute = brute_process(n, cons)
    if mine != brute:
        decisions_ok = False
        print(f"  decision mismatch: n={n} cons={cons}\n    mine ={mine}\n    brute={brute}")
        break
check("accept/reject decisions match brute BFS reference (500 constraint sequences)", decisions_ok)

# --- reported diffs match the brute offsets --------------------------------
rng = LCG(4242)
diff_ok = True
for _ in range(400):
    n = rng.randint(2, 8)
    cons = []
    dsu = WeightedDSU(n)
    for _ in range(rng.randint(1, 15)):
        x = rng.randint(0, n - 1)
        y = rng.randint(0, n - 1)
        dd = rng.randint(-8, 8)
        if dsu.union(x, y, dd):     # only keep accepted (consistent) constraints
            cons.append((x, y, dd))
    # now every accepted constraint's diff must be exactly the imposed value
    for (x, y, dd) in cons:
        if dsu.diff(x, y) != dd:
            diff_ok = False
            break
    # symmetry and transitivity: diff(x,y) = -diff(y,x); diff(x,z)=diff(x,y)+diff(y,z) if connected
    for _ in range(10):
        a, b, c = (rng.randint(0, n - 1) for _ in range(3))
        if dsu.connected(a, b) and dsu.diff(a, b) != -dsu.diff(b, a):
            diff_ok = False
            break
        if dsu.connected(a, b) and dsu.connected(b, c):
            if dsu.diff(a, c) != dsu.diff(a, b) + dsu.diff(b, c):
                diff_ok = False
                break
    if not diff_ok:
        break
check("reported differences are self-consistent (symmetry + transitivity) (400 sequences)", diff_ok)

# --- consistent constraints are always accepted ----------------------------
# build from a hidden ground-truth potential assignment: every constraint derived from it is consistent
rng = LCG(777)
consistent_ok = True
for _ in range(300):
    n = rng.randint(2, 10)
    truth = [rng.randint(-50, 50) for _ in range(n)]
    dsu = WeightedDSU(n)
    ok = True
    for _ in range(rng.randint(1, 20)):
        x = rng.randint(0, n - 1)
        y = rng.randint(0, n - 1)
        d = truth[x] - truth[y]          # a genuinely consistent constraint
        if not dsu.union(x, y, d):
            ok = False
            break
    if not ok:
        consistent_ok = False
        break
    # every diff must equal the ground-truth difference
    for x in range(n):
        for y in range(n):
            if dsu.connected(x, y) and dsu.diff(x, y) != truth[x] - truth[y]:
                consistent_ok = False
                break
        if not consistent_ok:
            break
    if not consistent_ok:
        break
check("constraints from a ground-truth assignment are all accepted and diffs match it", consistent_ok)

# --- parity DSU (bipartite / same-or-different) ----------------------------
p = ParityDSU(5)
check("relate same(0,1) accepted", p.relate(0, 1, True))
check("relate diff(1,2) accepted", p.relate(1, 2, False))
check("same_group(0,1) True", p.same_group(0, 1) is True)
check("same_group(0,2) False (0=1 same, 1-2 differ)", p.same_group(0, 2) is False)
check("same_group to unknown is None", p.same_group(0, 4) is None)
# a contradiction: force an odd cycle
p2 = ParityDSU(3)
p2.relate(0, 1, False)
p2.relate(1, 2, False)
check("odd cycle: relating 0 and 2 as different is a contradiction",
      not p2.relate(0, 2, False))
check("but relating 0 and 2 as same is consistent", p2.relate(0, 2, True))

# --- parity matches a mod-2 brute reference --------------------------------
rng = LCG(555)
parity_ok = True
for _ in range(300):
    n = rng.randint(2, 8)
    cons = []
    for _ in range(rng.randint(1, 15)):
        x = rng.randint(0, n - 1)
        y = rng.randint(0, n - 1)
        same = rng.rand() % 2 == 0
        cons.append((x, y, 0 if same else 1))
    p = ParityDSU(n)
    mine = [p.relate(x, y, dd == 0) for (x, y, dd) in cons]
    brute = brute_process(n, cons)          # d in {0,1}; brute checks equality mod nothing but works
    # note: brute uses exact equality; parity uses mod 2, so compare mod-2 consistency directly
    # reconstruct brute mod-2 decisions
    from collections import deque, defaultdict
    adj = defaultdict(list)
    brute2 = []
    for (x, y, dd) in cons:
        # BFS offset mod 2
        pot = {x: 0}
        q = deque([x])
        while q:
            u = q.popleft()
            for v, delta in adj[u]:
                if v not in pot:
                    pot[v] = (pot[u] + delta) % 2
                    q.append(v)
        known = pot.get(y)
        if known is None:
            adj[x].append((y, dd % 2))
            adj[y].append((x, (-dd) % 2))
            brute2.append(True)
        else:
            brute2.append(known == dd % 2)
    if mine != brute2:
        parity_ok = False
        break
check("parity DSU decisions match a mod-2 brute reference (300 sequences)", parity_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all weighted_dsu tests passed")

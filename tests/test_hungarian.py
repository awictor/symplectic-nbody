"""Tests for hungarian: optimal assignment vs brute-force permutations, min/max, rectangular."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hungarian import solve, min_cost, max_cost

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 2718
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_min(cost):
    n = len(cost)
    m = len(cost[0])
    best = float("inf")
    # assign each row to a distinct column (n <= m assumed for the brute check)
    for perm in itertools.permutations(range(m), n):
        total = sum(cost[i][perm[i]] for i in range(n))
        best = min(best, total)
    return best


def brute_max(cost):
    n = len(cost)
    m = len(cost[0])
    best = float("-inf")
    for perm in itertools.permutations(range(m), n):
        best = max(best, sum(cost[i][perm[i]] for i in range(n)))
    return best


# --- known hand-worked example ---------------------------------------------
cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
a, t = solve(cost)
check("known 3x3 optimal cost is 5", t == 5.0)
check("assignment is a permutation", sorted(a) == [0, 1, 2])
check("assignment cost matches the sum", sum(cost[i][a[i]] for i in range(3)) == t)

# --- matches brute force over many random square matrices ------------------
ok = True
for _ in range(60):
    n = 2 + int(rng() * 4)          # 2..5
    c = [[int(rng() * 20) for _ in range(n)] for _ in range(n)]
    _, got = solve(c)
    if got != brute_min(c):
        ok = False
        break
check("min assignment matches brute force over 60 random square matrices", ok)

# --- assignment returned is always a valid permutation ---------------------
ok = True
for _ in range(40):
    n = 2 + int(rng() * 4)
    c = [[int(rng() * 50) for _ in range(n)] for _ in range(n)]
    a, t = solve(c)
    if sorted(a) != list(range(n)):
        ok = False
        break
    if abs(sum(c[i][a[i]] for i in range(n)) - t) > 1e-9:
        ok = False
        break
check("assignment is always a valid permutation with matching cost", ok)

# --- maximization ----------------------------------------------------------
ok = True
for _ in range(40):
    n = 2 + int(rng() * 4)
    c = [[int(rng() * 20) for _ in range(n)] for _ in range(n)]
    _, got = solve(c, maximize=True)
    if got != brute_max(c):
        ok = False
        break
check("max assignment matches brute force", ok)

# --- identity / diagonal matrix -------------------------------------------
diag = [[0 if i == j else 10 for j in range(5)] for i in range(5)]
a, t = solve(diag)
check("diagonal-zero matrix picks the diagonal", t == 0 and a == [0, 1, 2, 3, 4])

# --- the row/column reduction invariant ------------------------------------
# subtracting a constant from a row shifts the optimum by that constant but keeps the same assignment
c = [[7, 2, 5], [3, 8, 1], [6, 4, 9]]
a1, t1 = solve(c)
c2 = [row[:] for row in c]
for j in range(3):
    c2[0][j] -= 3          # subtract 3 from row 0
a2, t2 = solve(c2)
check("row reduction preserves the optimal assignment", a1 == a2)
check("row reduction shifts the cost by exactly the subtracted amount", abs((t1 - 3) - t2) < 1e-9)

# --- rectangular matrices (fewer workers than jobs) ------------------------
rect = [[1, 2, 3], [4, 5, 6]]
a, t = solve(rect)
check("rectangular 2x3: cost is the best 2-of-3 assignment", t == brute_min(rect))
check("rectangular assignment uses distinct columns", len(set(a)) == 2)

# more jobs-than-workers, larger
rect2 = [[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8]]
a, t = solve(rect2)
check("rectangular 3x4 matches brute force", t == brute_min(rect2))

# --- single element --------------------------------------------------------
check("1x1 assignment", solve([[42]]) == ([0], 42))

# --- min_cost / max_cost convenience --------------------------------------
c = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
check("min_cost convenience", min_cost(c) == 5)
check("max_cost convenience", max_cost(c) == brute_max(c))

# --- a larger instance (n=8) still matches brute (8! = 40320, feasible) ----
c8 = [[int(rng() * 100) for _ in range(8)] for _ in range(8)]
_, got = solve(c8)
check("n=8 matches brute force", got == brute_min(c8))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all hungarian tests passed")

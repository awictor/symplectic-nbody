"""Tests for li_chao: lower/upper envelope queries vs brute force over all lines."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from li_chao import LiChaoTree, convex_hull_trick_dp, brute_min, brute_max

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
lines = [(2, 3), (-1, 10), (0, 5), (1, -2), (-3, 20)]
t = LiChaoTree(-100, 100)
for m, b in lines:
    t.add_line(m, b)
known_ok = all(abs(t.query(x) - brute_min(lines, x)) < 1e-9
               for x in [-50, -10, 0, 3, 7, 25, 80])
check("known line set: all queries match brute min", known_ok)

check("empty tree returns +inf for min", LiChaoTree(-10, 10).query(0) == float("inf"))
check("empty maximise tree returns -inf", LiChaoTree(-10, 10, maximize=True).query(0) == float("-inf"))

# single line
t = LiChaoTree(-100, 100)
t.add_line(3, 7)
check("single line queried correctly", abs(t.query(4) - (3 * 4 + 7)) < 1e-9)

# --- exhaustive min validation ---------------------------------------------
rng = LCG(2026)
min_ok = True
for _ in range(300):
    n = rng.randint(1, 25)
    lines = [(rng.randint(-20, 20), rng.randint(-100, 100)) for _ in range(n)]
    t = LiChaoTree(-1000, 1000)
    for m, b in lines:
        t.add_line(m, b)
    for _ in range(15):
        x = rng.randint(-1000, 1000)
        if abs(t.query(x) - brute_min(lines, x)) > 1e-6:
            min_ok = False
            print(f"  min mismatch x={x}: tree={t.query(x)} brute={brute_min(lines, x)}")
            break
    if not min_ok:
        break
check("minimum queries match brute force (300 line sets x 15 points)", min_ok)

# --- exhaustive max validation ---------------------------------------------
rng = LCG(4242)
max_ok = True
for _ in range(300):
    n = rng.randint(1, 25)
    lines = [(rng.randint(-20, 20), rng.randint(-100, 100)) for _ in range(n)]
    t = LiChaoTree(-1000, 1000, maximize=True)
    for m, b in lines:
        t.add_line(m, b)
    for _ in range(15):
        x = rng.randint(-1000, 1000)
        if abs(t.query(x) - brute_max(lines, x)) > 1e-6:
            max_ok = False
            break
    if not max_ok:
        break
check("maximum queries match brute force (300 line sets x 15 points)", max_ok)

# --- interleaved insert/query (the case a monotonic CHT can't handle) ------
rng = LCG(777)
interleave_ok = True
for _ in range(200):
    lines = []
    t = LiChaoTree(-500, 500)
    for _ in range(rng.randint(1, 20)):
        m, b = rng.randint(-15, 15), rng.randint(-50, 50)
        lines.append((m, b))
        t.add_line(m, b)
        # query right after each insert
        x = rng.randint(-500, 500)
        if abs(t.query(x) - brute_min(lines, x)) > 1e-6:
            interleave_ok = False
            break
    if not interleave_ok:
        break
check("interleaved inserts and queries stay correct (arbitrary order)", interleave_ok)

# --- adding a line never raises the minimum --------------------------------
rng = LCG(555)
monotone_ok = True
for _ in range(200):
    t = LiChaoTree(-200, 200)
    prev = float("inf")
    x = rng.randint(-200, 200)
    for _ in range(15):
        t.add_line(rng.randint(-10, 10), rng.randint(-40, 40))
        cur = t.query(x)
        if cur > prev + 1e-9:
            monotone_ok = False
            break
        prev = cur
    if not monotone_ok:
        break
check("adding lines never increases the minimum at a fixed x", monotone_ok)

# --- parallel lines: only the lowest intercept ever wins -------------------
t = LiChaoTree(-100, 100)
for b in [10, 3, 7, -5, 20]:
    t.add_line(2, b)         # all slope 2
check("parallel lines: minimum is the lowest intercept everywhere",
      all(abs(t.query(x) - (2 * x - 5)) < 1e-9 for x in [-50, 0, 33, 90]))

# --- real-valued x ---------------------------------------------------------
lines = [(1.5, 2.0), (-0.5, 8.0), (0.25, -1.0)]
t = LiChaoTree(-50, 50)
for m, b in lines:
    t.add_line(m, b)
check("real-valued slopes and query points work",
      all(abs(t.query(x) - brute_min(lines, x)) < 1e-9 for x in [-12.5, 0.0, 3.7, 41.2]))

# --- convex-hull-trick DP matches an O(n^2) reference ----------------------
rng = LCG(31337)
cht_ok = True
for _ in range(100):
    n = rng.randint(1, 15)
    costs = [rng.randint(-20, 20) for _ in range(n)]
    slopes = [rng.randint(-10, 10) for _ in range(n)]
    intercepts = [rng.randint(-30, 30) for _ in range(n)]
    got = convex_hull_trick_dp(costs, slopes, intercepts)
    # brute: dp[i] = min over j<=i of slopes[j]*costs[i] + intercepts[j]
    want = []
    for i in range(n):
        want.append(min(slopes[j] * costs[i] + intercepts[j] for j in range(i + 1)))
    if any(abs(g - w) > 1e-6 for g, w in zip(got, want)):
        cht_ok = False
        break
check("convex-hull-trick DP matches the O(n^2) reference (100 instances)", cht_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all li_chao tests passed")

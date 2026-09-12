"""Tests for interval_scheduling: weighted DP vs brute force, greedy count, compatibility."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from interval_scheduling import schedule, activity_selection, is_compatible, total_weight

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 4242
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_max_weight(jobs):
    n = len(jobs)
    best = 0.0
    for mask in range(1 << n):
        subset = [jobs[i] for i in range(n) if mask & (1 << i)]
        if is_compatible(subset):
            best = max(best, total_weight(subset))
    return best


def brute_max_count(jobs):
    n = len(jobs)
    best = 0
    for mask in range(1 << n):
        subset = [jobs[i] for i in range(n) if mask & (1 << i)]
        if is_compatible(subset):
            best = max(best, len(subset))
    return best


# --- known instance --------------------------------------------------------
jobs = [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4), (5, 8, 11), (7, 9, 2)]
w, chosen = schedule(jobs)
check("known instance max weight is 17", w == 17)
check("chosen jobs are compatible", is_compatible(chosen))
check("chosen jobs sum to the reported weight", total_weight(chosen) == w)

# --- DP matches brute force over random instances --------------------------
ok_w = ok_valid = True
for _ in range(200):
    n = 1 + int(rng() * 10)
    js = []
    for _ in range(n):
        s = int(rng() * 20)
        f = s + 1 + int(rng() * 10)
        wgt = 1 + int(rng() * 20)
        js.append((s, f, wgt))
    w, chosen = schedule(js)
    if w != brute_max_weight(js):
        ok_w = False
        break
    # chosen must be compatible and sum to w
    if not is_compatible(chosen) or abs(total_weight(chosen) - w) > 1e-9:
        ok_valid = False
        break
check("weighted DP matches brute force over 200 random instances", ok_w)
check("returned schedule is compatible and sums to the max weight", ok_valid)

# --- greedy count matches the maximum independent set ----------------------
ok = True
for _ in range(200):
    n = 1 + int(rng() * 10)
    js = [(int(rng() * 15), 0, 1) for _ in range(n)]
    js = [(s, s + 1 + int(rng() * 8), 1) for s, _, _ in js]
    c, sel = activity_selection(js)
    if c != brute_max_count(js):
        ok = False
        break
    if not is_compatible(sel):
        ok = False
        break
check("greedy activity selection matches the max independent set", ok)

# --- equal weights: weighted DP agrees with the greedy count --------------
ok = True
for _ in range(50):
    n = 1 + int(rng() * 8)
    js = [(s, s + 1 + int(rng() * 6), 1) for s in [int(rng() * 12) for _ in range(n)]]
    w, _ = schedule(js)          # each weight 1, so max weight == max count
    c, _ = activity_selection(js)
    if w != c:
        ok = False
        break
check("with unit weights, DP weight equals greedy count", ok)

# --- non-overlapping jobs are all selected ---------------------------------
disjoint = [(0, 1, 3), (2, 3, 5), (4, 5, 2), (6, 7, 8)]
w, chosen = schedule(disjoint)
check("disjoint jobs are all selected", w == 18 and len(chosen) == 4)

# --- a high-weight long job beats several small ones -----------------------
# one job [0,10] weight 100 vs five jobs [0,2],[2,4],... weight 10 each (total 50)
long_job = [(0, 10, 100)] + [(2 * i, 2 * i + 2, 10) for i in range(5)]
w, chosen = schedule(long_job)
check("high-weight long job chosen over many small ones", w == 100 and chosen == [(0, 10, 100)])

# --- touching intervals are compatible (finish == next start) --------------
touching = [(0, 5, 10), (5, 10, 10)]
w, chosen = schedule(touching)
check("touching intervals (finish == start) are both selected", w == 20 and len(chosen) == 2)

# --- overlapping intervals: only the better one -----------------------------
overlap = [(0, 10, 5), (0, 10, 8)]
w, chosen = schedule(overlap)
check("of two overlapping jobs, the higher-weight one is chosen", w == 8)

# --- single job and empty --------------------------------------------------
check("single job is selected", schedule([(0, 1, 7)]) == (7, [(0, 1, 7)]))
check("no jobs -> zero weight", schedule([]) == (0.0, []))
check("empty activity selection", activity_selection([]) == (0, []))

# --- a larger instance ------------------------------------------------------
big = [(int(rng() * 100), 0, 0)] * 0
big = []
for _ in range(500):
    s = int(rng() * 200)
    big.append((s, s + 1 + int(rng() * 20), 1 + int(rng() * 50)))
w, chosen = schedule(big)
check("large instance: schedule is compatible", is_compatible(chosen))
check("large instance: weight matches the chosen sum", abs(total_weight(chosen) - w) < 1e-9)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all interval_scheduling tests passed")

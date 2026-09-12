"""Tests for mo_algorithm: offline range distinct-count and power-sum vs brute recomputation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mo_algorithm import (range_distinct, range_power_sum, brute_distinct, brute_power_sum)

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
arr = [1, 1, 2, 3, 3, 3, 2, 1, 4]
queries = [(0, 3), (2, 6), (0, 8), (4, 4), (1, 7)]
check("distinct known case", range_distinct(arr, queries) == [3, 2, 4, 1, 3])
check("power-sum known case", range_power_sum(arr, queries) == [6, 13, 23, 1, 17])

check("empty query list returns empty", range_distinct([1, 2, 3], []) == [])
check("single element distinct is 1", range_distinct([5, 6, 7], [(1, 1)]) == [1])

# --- exhaustive vs brute: distinct -----------------------------------------
rng = LCG(2026)
distinct_ok = True
for _ in range(300):
    n = rng.randint(1, 40)
    vmax = rng.randint(1, 10)
    arr = [rng.randint(1, vmax) for _ in range(n)]
    q = rng.randint(1, 30)
    queries = []
    for _ in range(q):
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        queries.append((min(a, b), max(a, b)))
    if range_distinct(arr, queries) != brute_distinct(arr, queries):
        distinct_ok = False
        print(f"  distinct mismatch arr={arr} q={queries}")
        break
check("range distinct-count matches brute force (300 arrays x up to 30 queries)", distinct_ok)

# --- exhaustive vs brute: power sum ----------------------------------------
rng = LCG(4242)
power_ok = True
for _ in range(300):
    n = rng.randint(1, 40)
    vmax = rng.randint(1, 8)
    arr = [rng.randint(1, vmax) for _ in range(n)]
    q = rng.randint(1, 30)
    queries = []
    for _ in range(q):
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        queries.append((min(a, b), max(a, b)))
    if range_power_sum(arr, queries) != brute_power_sum(arr, queries):
        power_ok = False
        break
check("range power-sum matches brute force (300 arrays x up to 30 queries)", power_ok)

# --- answers respect original query order ----------------------------------
# shuffle-invariance: reordering queries reorders answers correspondingly
rng = LCG(777)
order_ok = True
for _ in range(100):
    n = rng.randint(2, 30)
    arr = [rng.randint(1, 6) for _ in range(n)]
    queries = []
    for _ in range(rng.randint(2, 15)):
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        queries.append((min(a, b), max(a, b)))
    ans = range_distinct(arr, queries)
    # each answer must equal the direct recomputation of the query at the SAME index
    for i, (l, r) in enumerate(queries):
        if ans[i] != len(set(arr[l:r + 1])):
            order_ok = False
            break
    if not order_ok:
        break
check("answers are returned in the original query order", order_ok)

# --- full-array and duplicate queries --------------------------------------
arr = [4, 4, 4, 4]
check("all-equal array: distinct is always 1", range_distinct(arr, [(0, 3), (1, 2)]) == [1, 1])
check("all-equal array: power sum is r*r for full window", range_power_sum(arr, [(0, 3)]) == [16])

arr = [7, 8, 9, 10]
check("all-distinct array: distinct equals window length",
      range_distinct(arr, [(0, 3), (1, 2)]) == [4, 2])

# repeated identical queries all get the same answer
arr = [1, 2, 2, 3, 1]
q = [(0, 4)] * 5
check("repeated queries all answered identically", range_distinct(arr, q) == [3, 3, 3, 3, 3])

# --- larger random stress (performance sanity) -----------------------------
rng = LCG(31337)
arr = [rng.randint(1, 50) for _ in range(2000)]
queries = []
for _ in range(2000):
    a = rng.randint(0, 1999)
    b = rng.randint(0, 1999)
    queries.append((min(a, b), max(a, b)))
mo = range_distinct(arr, queries)
# spot-check 40 of them against brute
spot_ok = all(mo[i] == len(set(arr[queries[i][0]:queries[i][1] + 1]))
              for i in range(0, 2000, 50))
check("2000-element array, 2000 queries: spot-checks match brute", spot_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all mo_algorithm tests passed")

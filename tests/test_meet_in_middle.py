"""Tests for meet_in_middle: subset-sum / closest / count vs brute-force enumeration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from meet_in_middle import (subset_sum_exists, max_subset_sum_under, closest_subset_sum,
                            count_subsets_with_sum, brute_subset_sum_exists, brute_max_under,
                            brute_closest, brute_count_with_sum)

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


# --- known cases ------------------------------------------------------------
items = [3, 34, 4, 12, 5, 2]
check("subset summing to 9 exists (4+5)", subset_sum_exists(items, 9))
check("no subset sums to 30", not subset_sum_exists(items, 30))
check("empty subset sums to 0", subset_sum_exists([1, 2, 3], 0))
check("max subset sum under 10 is 10 (a subset hits it)", max_subset_sum_under(items, 10) == 10)
check("closest achievable sum to 20 is 20", closest_subset_sum(items, 20) == 20)
check("two subsets sum to 9", count_subsets_with_sum(items, 9) == 2)
check("one subset (empty) sums to 0", count_subsets_with_sum([1, 2, 3], 0) == 1)

# --- huge values where a pseudo-poly DP would be infeasible ----------------
big = [10 ** 15, 3 * 10 ** 14, 7 * 10 ** 14, 2 * 10 ** 15]
check("meet-in-the-middle handles astronomically large values",
      subset_sum_exists(big, 10 ** 15 + 7 * 10 ** 14))
check("no subset for an impossible huge target",
      not subset_sum_exists(big, 10 ** 15 + 1))

# --- subset-sum existence vs brute -----------------------------------------
rng = LCG(2026)
exists_ok = True
for _ in range(400):
    n = rng.randint(0, 14)
    items = [rng.randint(1, 40) for _ in range(n)]
    # test a mix of reachable and random targets
    reachable = sum(items[i] for i in range(n) if rng.rand() % 2)
    for target in (reachable, rng.randint(0, 40 * max(1, n))):
        if subset_sum_exists(items, target) != brute_subset_sum_exists(items, target):
            exists_ok = False
            print(f"  exists mismatch: items={items} target={target}")
            break
    if not exists_ok:
        break
check("subset-sum existence matches brute force (400 instances)", exists_ok)

# --- max-under-capacity vs brute -------------------------------------------
rng = LCG(4242)
max_ok = True
for _ in range(400):
    n = rng.randint(0, 14)
    items = [rng.randint(1, 30) for _ in range(n)]
    cap = rng.randint(0, 30 * max(1, n))
    if max_subset_sum_under(items, cap) != brute_max_under(items, cap):
        max_ok = False
        print(f"  max-under mismatch: items={items} cap={cap}")
        break
check("max subset sum under capacity matches brute force (400 instances)", max_ok)

# --- closest subset sum vs brute -------------------------------------------
rng = LCG(777)
closest_ok = True
for _ in range(400):
    n = rng.randint(0, 13)
    items = [rng.randint(1, 30) for _ in range(n)]
    target = rng.randint(0, 30 * max(1, n))
    mm = closest_subset_sum(items, target)
    bf = brute_closest(items, target)
    # the closest DISTANCE must match (ties may pick different equal-distance sums, but our tie rule
    # is deterministic toward the smaller, so the values should match too)
    if mm != bf:
        closest_ok = False
        print(f"  closest mismatch: items={items} target={target} mm={mm} bf={bf}")
        break
check("closest achievable subset sum matches brute force (400 instances)", closest_ok)

# --- count subsets with exact sum vs brute ---------------------------------
rng = LCG(555)
count_ok = True
for _ in range(400):
    n = rng.randint(0, 13)
    items = [rng.randint(1, 12) for _ in range(n)]
    target = rng.randint(0, 12 * max(1, n))
    if count_subsets_with_sum(items, target) != brute_count_with_sum(items, target):
        count_ok = False
        print(f"  count mismatch: items={items} target={target}")
        break
check("subset-count with exact sum matches brute force (400 instances)", count_ok)

# --- total subsets: counts over all targets sum to 2^n ---------------------
rng = LCG(99)
total_ok = True
for _ in range(100):
    n = rng.randint(0, 12)
    items = [rng.randint(1, 6) for _ in range(n)]
    total = sum(count_subsets_with_sum(items, t) for t in range(sum(items) + 1))
    if total != (1 << n):
        total_ok = False
        break
check("subset counts over all sums total 2^n", total_ok)

# --- closest sum is always achievable --------------------------------------
rng = LCG(31337)
achievable_ok = True
for _ in range(200):
    n = rng.randint(1, 12)
    items = [rng.randint(1, 20) for _ in range(n)]
    target = rng.randint(0, 200)
    mm = closest_subset_sum(items, target)
    if not subset_sum_exists(items, mm):
        achievable_ok = False
        break
check("the closest subset sum is itself achievable", achievable_ok)

# --- a larger instance (n=30) infeasible for brute but fast here -----------
rng = LCG(2718)
items = [rng.randint(1, 10 ** 9) for _ in range(30)]
# construct a definitely-reachable target from a known subset
known = sum(items[i] for i in range(0, 30, 3))
check("n=30 with billion-scale values: known subset sum is found",
      subset_sum_exists(items, known))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all meet_in_middle tests passed")

"""Tests for knapsack.py -- 0/1 knapsack and its relatives.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The DP is proven correct
exhaustively against a brute-force subset search.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import knapsack as K  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- classic example --------------------------------------------------------
w, v, cap = [1, 3, 4, 5], [1, 4, 5, 7], 7
val, items = K.knapsack(w, v, cap)
check("classic knapsack optimal value is 9", val == 9)
check("chosen items fit within capacity", sum(w[i] for i in items) <= cap)
check("chosen items sum to the optimal value", sum(v[i] for i in items) == val)
check("value-only version agrees", K.knapsack_value(w, v, cap) == val)

# --- edge cases -------------------------------------------------------------
check("zero capacity takes nothing", K.knapsack([1, 2], [5, 6], 0) == (0, []))
check("no items gives zero value", K.knapsack([], [], 10) == (0, []))
check("everything fits takes all items", K.knapsack([1, 2, 3], [1, 1, 1], 10)[0] == 3)
check("an item heavier than capacity is skipped", K.knapsack([5], [10], 3) == (0, []))
try:
    K.knapsack([1, 2], [1], 5)
    check("rejects mismatched lengths", False)
except ValueError:
    check("rejects mismatched lengths", True)
try:
    K.knapsack([1], [1], -1)
    check("rejects negative capacity", False)
except ValueError:
    check("rejects negative capacity", True)

# --- exhaustive check against brute force ----------------------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 16


gen = lcg(7)


def rnd(lo, hi):
    return lo + next(gen) % (hi - lo + 1)


mismatch = 0
recon_bad = 0
for _ in range(1000):
    n = rnd(0, 12)
    cap = rnd(0, 30)
    ws = [rnd(1, 12) for _ in range(n)]
    vs = [rnd(1, 25) for _ in range(n)]
    dv, items = K.knapsack(ws, vs, cap)
    bv, _ = K.brute_knapsack(ws, vs, cap)
    if dv != bv or K.knapsack_value(ws, vs, cap) != bv:
        mismatch += 1
    # the reconstructed item set must be feasible and achieve the optimum
    if sum(ws[i] for i in items) > cap or sum(vs[i] for i in items) != dv:
        recon_bad += 1
check("EVERY one of 1000 random knapsacks matches brute force", mismatch == 0)
check("the reconstructed item set is always feasible and optimal", recon_bad == 0)

# --- subset sum -------------------------------------------------------------
nums = [3, 34, 4, 12, 5, 2]
s = K.subset_sum(nums, 9)
check("subset_sum finds a subset summing to the target", s is not None and sum(nums[i] for i in s) == 9)
check("subset_sum returns None when impossible", K.subset_sum([2, 4, 6], 5) is None)
check("subset_sum hits 0 with the empty set", K.subset_sum([1, 2, 3], 0) == [])
check("subset_sum can use the whole set", sum(nums[i] for i in (K.subset_sum(nums, sum(nums)) or [])) == sum(nums))
check("subset_sum rejects a negative target", K.subset_sum([1, 2], -3) is None)
# exhaustive subset-sum check
sub_bad = 0
for _ in range(500):
    n = rnd(0, 12)
    nums = [rnd(1, 15) for _ in range(n)]
    target = rnd(0, 40)
    got = K.subset_sum(nums, target)
    # brute: does any subset hit target?
    possible = any(
        sum(nums[i] for i in range(n) if mask & (1 << i)) == target
        for mask in range(1 << n)
    )
    if (got is not None) != possible:
        sub_bad += 1
    elif got is not None and sum(nums[i] for i in got) != target:
        sub_bad += 1
check("subset_sum matches brute force over 500 random cases", sub_bad == 0)

# --- unbounded knapsack -----------------------------------------------------
check("unbounded knapsack reuses items", K.unbounded_knapsack([1, 3, 4], [1, 4, 5], 7) == 9)
check("unbounded >= 0/1 value (reuse can only help)",
      K.unbounded_knapsack([2, 3], [3, 4], 12) >= K.knapsack_value([2, 3], [3, 4], 12))
check("unbounded with one item is floor(cap/w) copies",
      K.unbounded_knapsack([3], [5], 10) == (10 // 3) * 5)
check("unbounded zero capacity is 0", K.unbounded_knapsack([1, 2], [1, 1], 0) == 0)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall knapsack tests passed")

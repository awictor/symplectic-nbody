"""Tests for fenwick.py -- the Fenwick / binary indexed tree.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Every operation is cross-checked
against a brute-force array over a deterministic pseudo-random workload.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fenwick  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- construction -----------------------------------------------------------
vals = [3, 1, 4, 1, 5, 9, 2, 6]
ft = fenwick.FenwickTree.from_values(vals)
check("from_values reconstructs the array", ft.to_list() == vals)
check("total equals the array sum", ft.total() == sum(vals))
check("empty tree has total 0", fenwick.FenwickTree(0).total() == 0)
check("a fresh tree is all zeros", fenwick.FenwickTree(5).to_list() == [0, 0, 0, 0, 0])
try:
    fenwick.FenwickTree(-1)
    check("rejects negative size", False)
except ValueError:
    check("rejects negative size", True)

# --- prefix and range sums against brute force -----------------------------
def prefix_brute(a, count):
    return sum(a[:count])

ok = all(ft.prefix_sum(c) == prefix_brute(vals, c) for c in range(len(vals) + 1))
check("prefix_sum matches brute force for all counts", ok)
check("prefix_sum(0) is 0", ft.prefix_sum(0) == 0)
check("prefix_sum(n) is the total", ft.prefix_sum(len(vals)) == sum(vals))
ok = all(ft.range_sum(lo, hi) == sum(vals[lo:hi])
         for lo in range(len(vals) + 1) for hi in range(lo, len(vals) + 1))
check("range_sum matches brute force for all ranges", ok)
check("get returns each element", [ft.get(i) for i in range(len(vals))] == vals)
try:
    ft.prefix_sum(99)
    check("prefix_sum rejects out-of-range count", False)
except IndexError:
    check("prefix_sum rejects out-of-range count", True)

# --- point updates track a brute-force array -------------------------------
brute = list(vals)
ft2 = fenwick.FenwickTree.from_values(vals)
# a deterministic sequence of updates
state = 12345
for _ in range(300):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    idx = (state >> 16) % len(vals)
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    delta = ((state >> 16) % 21) - 10  # -10..10
    ft2.update(idx, delta)
    brute[idx] += delta
check("point updates keep the tree in sync with brute force", ft2.to_list() == brute)
check("prefix sums stay correct after many updates",
      all(ft2.prefix_sum(c) == sum(brute[:c]) for c in range(len(brute) + 1)))
try:
    ft2.update(99, 1)
    check("update rejects out-of-range index", False)
except IndexError:
    check("update rejects out-of-range index", True)

# --- set() ------------------------------------------------------------------
ft3 = fenwick.FenwickTree.from_values(vals)
ft3.set(2, 100)
check("set changes one element", ft3.get(2) == 100)
check("set leaves other elements untouched",
      [ft3.get(i) for i in range(len(vals)) if i != 2] == [vals[i] for i in range(len(vals)) if i != 2])
check("set updates the total correctly", ft3.total() == sum(vals) - vals[2] + 100)

# --- find_prefix (cumulative binary search) --------------------------------
nonneg = [2, 0, 3, 1, 4, 0, 5]  # prefix sums: 2,2,5,6,10,10,15
ftp = fenwick.FenwickTree.from_values(nonneg)


def find_brute(a, target):
    """Smallest index k with sum(a[:k+1]) >= target, or n if unreachable."""
    s = 0
    for k in range(len(a)):
        s += a[k]
        if s >= target:
            return k
    return len(a)


ok = all(ftp.find_prefix(t) == find_brute(nonneg, t) for t in range(0, 17))
check("find_prefix matches brute force across targets", ok)
check("find_prefix(1) skips leading zeros correctly", ftp.find_prefix(3) == 2)
check("unreachable target returns n", ftp.find_prefix(999) == len(nonneg))
check("target 0 returns index 0", ftp.find_prefix(0) == 0)

# --- larger randomized cross-check -----------------------------------------
n = 200
init = [(7 * i * i + 3) % 50 for i in range(n)]
big = fenwick.FenwickTree.from_values(init)
bb = list(init)
state = 999
mismatch = False
for _ in range(2000):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    op = (state >> 16) % 3
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    i = (state >> 16) % n
    if op == 0:  # update
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        d = ((state >> 16) % 21) - 10
        big.update(i, d)
        bb[i] += d
    elif op == 1:  # prefix
        if big.prefix_sum(i) != sum(bb[:i]):
            mismatch = True
            break
    else:  # range
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        j = (state >> 16) % n
        lo, hi = min(i, j), max(i, j)
        if big.range_sum(lo, hi) != sum(bb[lo:hi]):
            mismatch = True
            break
check("2000 mixed ops stay consistent with brute force on n=200", not mismatch)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall fenwick tests passed")

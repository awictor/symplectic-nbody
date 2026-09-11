"""Tests for quickselect.py -- k-th smallest selection.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Every result is cross-checked
against a full sort, exhaustively over random arrays.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import quickselect as Q  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


data = [7, 2, 9, 1, 5, 8, 3, 6, 4, 0]
srt = sorted(data)

# --- quickselect matches sorted for every rank ----------------------------
check("quickselect matches sorted for all k", all(Q.quickselect(data, k) == srt[k] for k in range(len(data))))
check("k=0 is the minimum", Q.quickselect(data, 0) == min(data))
check("k=n-1 is the maximum", Q.quickselect(data, len(data) - 1) == max(data))
check("does not modify the input", (Q.quickselect(data, 3), data == [7, 2, 9, 1, 5, 8, 3, 6, 4, 0])[1])
check("single element selects itself", Q.quickselect([42], 0) == 42)
try:
    Q.quickselect(data, 99)
    check("rejects out-of-range k", False)
except IndexError:
    check("rejects out-of-range k", True)

# --- median-of-medians matches sorted (worst-case linear) ------------------
check("median-of-medians matches sorted for all k",
      all(Q.median_of_medians_select(data, k) == srt[k] for k in range(len(data))))
check("MoM on a small list works", Q.median_of_medians_select([3, 1, 2], 1) == 2)
check("MoM handles a list of exactly 5", Q.median_of_medians_select([5, 4, 3, 2, 1], 2) == 3)
check("MoM handles more than 5 (recursion)", Q.median_of_medians_select(list(range(20, 0, -1)), 9) == 10)

# --- 1-indexed wrappers -----------------------------------------------------
check("kth_smallest(1) is the minimum", Q.kth_smallest(data, 1) == min(data))
check("kth_largest(1) is the maximum", Q.kth_largest(data, 1) == max(data))
check("kth_smallest(n) is the maximum", Q.kth_smallest(data, len(data)) == max(data))
check("kth_largest(2) is the second largest", Q.kth_largest(data, 2) == srt[-2])
try:
    Q.kth_largest(data, 0)
    check("kth_largest rejects k=0", False)
except IndexError:
    check("kth_largest rejects k=0", True)

# --- median -----------------------------------------------------------------
check("median of odd count", Q.median([3, 1, 2]) == 2)
check("median of even count averages the middle two", Q.median([1, 2, 3, 4]) == 2.5)
check("median of a single element", Q.median([7]) == 7)
check("median matches the sorted middle",
      Q.median(data) == (srt[4] + srt[5]) / 2)
try:
    Q.median([])
    check("median of empty raises", False)
except ValueError:
    check("median of empty raises", True)

# --- percentile -------------------------------------------------------------
check("0th percentile is the minimum", Q.percentile(data, 0) == min(data))
check("100th percentile is the maximum", Q.percentile(data, 100) == max(data))
check("50th percentile is a middle value", Q.percentile(data, 50) in srt)
check("percentiles are monotonic", Q.percentile(data, 25) <= Q.percentile(data, 75))
try:
    Q.percentile(data, 150)
    check("percentile rejects p out of range", False)
except ValueError:
    check("percentile rejects p out of range", True)

# --- duplicates -------------------------------------------------------------
check("quickselect handles all-equal data", Q.quickselect([5, 5, 5, 5], 2) == 5)
check("MoM handles duplicates", Q.median_of_medians_select([5, 5, 1, 5, 5], 0) == 1)
dupes = [3, 1, 3, 1, 2, 2, 3, 1]
check("quickselect matches sorted with duplicates",
      all(Q.quickselect(dupes, k) == sorted(dupes)[k] for k in range(len(dupes))))

# --- exhaustive cross-check against sorting --------------------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 16


gen = lcg(11)


def rnd(lo, hi):
    return lo + next(gen) % (hi - lo + 1)


qbad = mbad = 0
for _ in range(1000):
    n = rnd(1, 70)
    arr = [rnd(-40, 40) for _ in range(n)]
    s = sorted(arr)
    for k in (0, n // 2, n - 1, rnd(0, n - 1)):
        if Q.quickselect(arr, k, seed=rnd(1, 9999)) != s[k]:
            qbad += 1
        if Q.median_of_medians_select(arr, k) != s[k]:
            mbad += 1
check("quickselect matches sorting on 1000 random arrays (varied seeds)", qbad == 0)
check("median-of-medians matches sorting on 1000 random arrays", mbad == 0)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall quickselect tests passed")

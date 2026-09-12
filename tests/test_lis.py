"""Tests for lis: length + subsequence vs brute force and the O(n^2) DP, variants, edge cases."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lis import lis, lis_length, lis_dp, longest_decreasing_subsequence

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 246
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_lis(seq, strict=True):
    best = 0
    n = len(seq)
    for mask in range(1 << n):
        sub = [seq[i] for i in range(n) if mask & (1 << i)]
        ok = all((sub[i] < sub[i + 1]) if strict else (sub[i] <= sub[i + 1])
                 for i in range(len(sub) - 1))
        if ok:
            best = max(best, len(sub))
    return best


# --- known case ------------------------------------------------------------
length, sub = lis([10, 9, 2, 5, 3, 7, 101, 18])
check("classic LIS length is 4", length == 4)
check("witness is strictly increasing", all(sub[i] < sub[i + 1] for i in range(len(sub) - 1)))
check("witness has the reported length", len(sub) == length)

# --- length matches the O(n^2) DP -----------------------------------------
ok = True
for _ in range(200):
    n = int(rng() * 15)
    seq = [int(rng() * 20) for _ in range(n)]
    if lis_length(seq) != lis_dp(seq):
        ok = False
        break
check("patience-sort length matches the O(n^2) DP over 200 random sequences", ok)

# --- length matches brute force (short sequences) --------------------------
ok = True
for _ in range(300):
    n = int(rng() * 12)
    seq = [int(rng() * 15) for _ in range(n)]
    if lis_length(seq, strict=True) != brute_lis(seq, strict=True):
        ok = False
        break
check("strict LIS length matches brute force over 300 sequences", ok)

# --- the returned subsequence is genuine and optimal-length ----------------
ok = True
for _ in range(200):
    n = int(rng() * 14)
    seq = [int(rng() * 20) for _ in range(n)]
    length, sub = lis(seq)
    # is it strictly increasing?
    if any(sub[i] >= sub[i + 1] for i in range(len(sub) - 1)):
        ok = False
        break
    # is it a subsequence of seq (in order)?
    it = iter(seq)
    if not all(any(x == y for y in it) for x in sub):
        # re-check subsequence property properly
        pass
    # proper subsequence check
    idx = 0
    for x in sub:
        while idx < len(seq) and seq[idx] != x:
            idx += 1
        if idx >= len(seq):
            ok = False
            break
        idx += 1
    if not ok:
        break
    # optimal length
    if length != brute_lis(seq):
        ok = False
        break
check("returned subsequence is genuine, ordered, and optimal-length", ok)

# --- non-decreasing (weak) variant -----------------------------------------
ok = True
for _ in range(200):
    n = int(rng() * 12)
    seq = [int(rng() * 8) for _ in range(n)]     # small alphabet -> many equal elements
    if lis_length(seq, strict=False) != brute_lis(seq, strict=False):
        ok = False
        break
check("non-decreasing LIS matches brute force", ok)

# weak allows equal elements: [1,3,3,5] has a length-4 non-decreasing subsequence
check("non-decreasing counts equal elements", lis_length([1, 3, 3, 5], strict=False) == 4)
check("strict excludes equal elements", lis_length([1, 3, 3, 5], strict=True) == 3)

# --- longest decreasing subsequence ---------------------------------------
ld, ds = longest_decreasing_subsequence([9, 4, 3, 2, 5, 4, 3, 2])
check("LDS is strictly decreasing", all(ds[i] > ds[i + 1] for i in range(len(ds) - 1)))
check("LDS length matches LIS of the reversed comparison",
      ld == lis_length([-x for x in [9, 4, 3, 2, 5, 4, 3, 2]]))

# --- edge cases ------------------------------------------------------------
check("empty sequence: length 0", lis([]) == (0, []))
check("single element: length 1", lis([42]) == (1, [42]))
check("sorted ascending: LIS is the whole thing", lis_length([1, 2, 3, 4, 5]) == 5)
check("sorted descending: strict LIS is 1", lis_length([5, 4, 3, 2, 1]) == 1)
check("all equal: strict LIS is 1", lis_length([7, 7, 7, 7]) == 1)
check("all equal: non-decreasing LIS is n", lis_length([7, 7, 7, 7], strict=False) == 4)

# --- a permutation: LIS length relates to sortedness -----------------------
# the identity permutation has LIS = n
check("identity permutation LIS is n", lis_length(list(range(20))) == 20)
# a reversal has LIS = 1
check("reversed permutation LIS is 1", lis_length(list(range(20))[::-1]) == 1)

# --- larger random: fast version agrees with DP ----------------------------
big = [int(rng() * 1000) for _ in range(500)]
check("500-element sequence: patience sort == DP", lis_length(big) == lis_dp(big))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all lis tests passed")

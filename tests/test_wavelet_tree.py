"""Tests for wavelet_tree: access/rank/select/quantile/range_count vs brute force."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wavelet_tree import WaveletTree

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 31
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- access reconstructs the sequence --------------------------------------
seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
wt = WaveletTree(seq)
check("access reproduces the sequence", [wt.access(i) for i in range(len(seq))] == seq)

# --- rank matches a prefix count -------------------------------------------
rank_ok = True
for v in set(seq):
    for i in range(len(seq) + 1):
        if wt.rank(v, i) != seq[:i].count(v):
            rank_ok = False
check("rank matches prefix count for all values and prefixes", rank_ok)
check("rank of absent value is 0", wt.rank(100, len(seq)) == 0)

# --- select matches an occurrence scan -------------------------------------
select_ok = True
for v in set(seq):
    occurrences = [i for i, x in enumerate(seq) if x == v]
    for j, pos in enumerate(occurrences):
        if wt.select(v, j) != pos:
            select_ok = False
    # one past the last occurrence -> -1
    if wt.select(v, len(occurrences)) != -1:
        select_ok = False
check("select matches occurrence positions", select_ok)
check("select of absent value is -1", wt.select(100, 0) == -1)

# --- rank/select are inverse ----------------------------------------------
inv_ok = True
for v in set(seq):
    cnt = seq.count(v)
    for j in range(cnt):
        pos = wt.select(v, j)
        # the number of v's in seq[:pos+1] should be j+1
        if wt.rank(v, pos + 1) != j + 1:
            inv_ok = False
check("rank and select are consistent inverses", inv_ok)

# --- quantile matches sorted-slice lookup ----------------------------------
quant_ok = True
for _ in range(200):
    a = int(rng() * len(seq))
    b = int(rng() * len(seq))
    lo, hi = min(a, b), max(a, b)
    if lo == hi:
        continue
    k = int(rng() * (hi - lo))
    got = wt.quantile(lo, hi, k)
    want = sorted(seq[lo:hi])[k]
    if got != want:
        quant_ok = False
        break
check("quantile matches the k-th smallest of the sorted slice", quant_ok)

# quantile extremes: k=0 is the min, k=len-1 is the max of the range
check("quantile k=0 is the range minimum", wt.quantile(0, len(seq), 0) == min(seq))
check("quantile k=n-1 is the range maximum",
      wt.quantile(0, len(seq), len(seq) - 1) == max(seq))

# --- range_count matches a filtered scan -----------------------------------
rc_ok = True
for _ in range(200):
    a = int(rng() * (len(seq) + 1))
    b = int(rng() * (len(seq) + 1))
    lo, hi = min(a, b), max(a, b)
    vlo = int(rng() * 10)
    vhi = int(rng() * 10)
    if vhi < vlo:
        vlo, vhi = vhi, vlo
    got = wt.range_count(lo, hi, vlo, vhi)
    want = sum(1 for p in range(lo, hi) if vlo <= seq[p] <= vhi)
    if got != want:
        rc_ok = False
        break
check("range_count matches a filtered scan", rc_ok)

# --- big randomized cross-check --------------------------------------------
big = [int(rng() * 50) for _ in range(300)]
wtb = WaveletTree(big)
check("big: access reproduces sequence", [wtb.access(i) for i in range(len(big))] == big)

# random rank/select/quantile/range_count checks
ok_all = True
for _ in range(300):
    op = int(rng() * 4)
    if op == 0:      # rank
        v = int(rng() * 50)
        i = int(rng() * (len(big) + 1))
        if wtb.rank(v, i) != big[:i].count(v):
            ok_all = False; break
    elif op == 1:    # select
        v = int(rng() * 50)
        occ = [p for p, x in enumerate(big) if x == v]
        if occ:
            j = int(rng() * len(occ))
            if wtb.select(v, j) != occ[j]:
                ok_all = False; break
    elif op == 2:    # quantile
        a = int(rng() * len(big)); b = int(rng() * len(big))
        lo, hi = min(a, b), max(a, b)
        if lo < hi:
            k = int(rng() * (hi - lo))
            if wtb.quantile(lo, hi, k) != sorted(big[lo:hi])[k]:
                ok_all = False; break
    else:            # range_count
        a = int(rng() * (len(big) + 1)); b = int(rng() * (len(big) + 1))
        lo, hi = min(a, b), max(a, b)
        vlo = int(rng() * 50); vhi = int(rng() * 50)
        if vhi < vlo:
            vlo, vhi = vhi, vlo
        if wtb.range_count(lo, hi, vlo, vhi) != sum(1 for p in range(lo, hi) if vlo <= big[p] <= vhi):
            ok_all = False; break
check("randomized rank/select/quantile/range_count all match brute force", ok_all)

# --- single-value sequence -------------------------------------------------
uni = WaveletTree([7, 7, 7, 7])
check("uniform sequence access", [uni.access(i) for i in range(4)] == [7, 7, 7, 7])
check("uniform sequence rank", uni.rank(7, 3) == 3)
check("uniform sequence select", uni.select(7, 2) == 2)
check("uniform sequence quantile", uni.quantile(0, 4, 2) == 7)

# --- single element --------------------------------------------------------
one = WaveletTree([42])
check("single element access", one.access(0) == 42)
check("single element rank", one.rank(42, 1) == 1)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all wavelet_tree tests passed")

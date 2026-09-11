"""Tests for segment_tree: range query/update for sum/min/max, brute-force agreement, edges."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segment_tree import SegmentTree

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- sum aggregate ---------------------------------------------------------
st = SegmentTree([1, 2, 3, 4, 5], "sum")
check("full-range sum", st.query(0, 4) == 15)
check("sub-range sum", st.query(1, 3) == 9)
check("single-element query", st.query(2, 2) == 3)
st.range_add(1, 3, 10)                         # -> [1, 12, 13, 14, 5]
check("range-add updates the sum", st.query(0, 4) == 45)
check("range-add reflected element-wise", st.to_list() == [1, 12, 13, 14, 5])
check("query outside the updated range", st.query(0, 0) == 1 and st.query(4, 4) == 5)

# --- point update ----------------------------------------------------------
st.point_update(0, 100)
check("point update sets the value", st.to_list() == [100, 12, 13, 14, 5])
check("point update reflected in sum", st.query(0, 4) == 144)

# --- min aggregate ---------------------------------------------------------
mn = SegmentTree([5, 2, 8, 1, 9], "min")
check("min over all", mn.query(0, 4) == 1)
check("min over a sub-range", mn.query(0, 2) == 2)
mn.range_add(3, 4, 10)                         # 1->11, 9->19 ; min now 2
check("range-add shifts the min", mn.query(0, 4) == 2)

# --- max aggregate ---------------------------------------------------------
mx = SegmentTree([5, 2, 8, 1, 9], "max")
check("max over all", mx.query(0, 4) == 9)
check("max over a sub-range", mx.query(0, 1) == 5)
mx.range_add(0, 1, 100)                        # 5->105, 2->102 ; max now 105
check("range-add shifts the max", mx.query(0, 4) == 105)

# --- brute-force agreement over random mixed operations, all three aggregates
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


for agg, reduce_fn in [("sum", sum), ("min", min), ("max", max)]:
    arr = [int(rng() * 100) for _ in range(40)]
    tree = SegmentTree(arr, agg)
    ref = list(arr)
    ok = True
    for _ in range(3000):
        l = int(rng() * 40)
        r = int(rng() * 40)
        if l > r:
            l, r = r, l
        if rng() < 0.5:
            add = int(rng() * 20) - 10
            tree.range_add(l, r, add)
            for i in range(l, r + 1):
                ref[i] += add
        else:
            if tree.query(l, r) != reduce_fn(ref[l:r + 1]):
                ok = False
                break
    check(f"{agg} matches brute force over 3000 ops", ok)
    check(f"{agg} final array matches reference", tree.to_list() == ref)

# --- point updates match a plain list --------------------------------------
pt = SegmentTree([0] * 20, "sum")
plain = [0] * 20
for _ in range(500):
    i = int(rng() * 20)
    v = int(rng() * 50)
    pt.point_update(i, v)
    plain[i] = v
check("point updates match a plain list", pt.to_list() == plain)
check("point-updated sum correct", pt.query(0, 19) == sum(plain))

# --- edge ranges -----------------------------------------------------------
single = SegmentTree([42], "sum")
check("single-element tree query", single.query(0, 0) == 42)
single.range_add(0, 0, 8)
check("single-element range-add", single.query(0, 0) == 50)

# --- full-array range-add on a fresh tree ----------------------------------
full = SegmentTree([1, 1, 1, 1], "sum")
full.range_add(0, 3, 5)
check("full-array range-add", full.query(0, 3) == 24 and full.to_list() == [6, 6, 6, 6])

# --- overlapping range-adds accumulate -------------------------------------
ov = SegmentTree([0, 0, 0, 0, 0], "sum")
ov.range_add(0, 2, 3)     # [3,3,3,0,0]
ov.range_add(1, 4, 5)     # [3,8,8,5,5]
check("overlapping range-adds accumulate", ov.to_list() == [3, 8, 8, 5, 5])
check("overlap total sum", ov.query(0, 4) == 29)

# --- min identity on nested queries ----------------------------------------
mn2 = SegmentTree([4, 4, 4, 4], "min")
mn2.range_add(1, 2, -10)
check("min finds the lowered element", mn2.query(0, 3) == -6)
check("min of the untouched tail", mn2.query(3, 3) == 4)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all segment_tree tests passed")

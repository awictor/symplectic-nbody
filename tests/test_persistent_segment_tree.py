"""Tests for persistent segment tree: version prefix sums + range k-th smallest vs brute force."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from persistent_segment_tree import (  # noqa: E402
    PersistentSegmentTree,
    RangeKth,
    brute_kth_smallest,
)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def main():
    # ---- 1. persistent versions keep independent prefix sums ----------------------------
    rng = _lcg(2024)
    mism = 0
    for _ in range(100):
        size = 2 + rng() % 10
        pst = PersistentSegmentTree(size)
        shadow = [[0] * size]  # shadow[v] = array state at version v
        cur = 0
        for _ in range(15):
            pos = rng() % size
            delta = (rng() % 7) - 3
            cur = pst.update(cur, pos, delta)
            new_state = list(shadow[-1])
            new_state[pos] += delta
            shadow.append(new_state)
        # every version must reproduce every range sum
        for v in range(len(shadow)):
            for _ in range(5):
                a = rng() % size
                b = rng() % size
                lo, hi = min(a, b), max(a, b)
                got = pst.query(v, lo, hi)
                exp = sum(shadow[v][lo:hi + 1])
                if got != exp:
                    mism += 1
    check("all versions reproduce correct prefix/range sums", mism == 0, f"{mism}")

    # ---- 2. old versions remain valid after later updates -------------------------------
    pst = PersistentSegmentTree(5)
    v1 = pst.update(0, 2, 10)
    v2 = pst.update(v1, 4, 7)
    v3 = pst.update(v2, 2, -3)
    check("version 1 query unchanged by later updates", pst.query(v1, 0, 4) == 10)
    check("version 2 sum = 17", pst.query(v2, 0, 4) == 17)
    check("version 3 sum = 14", pst.query(v3, 0, 4) == 14)
    check("empty version 0 is zero", pst.query(0, 0, 4) == 0)

    # ---- 3. range k-th smallest matches sorting, all k ----------------------------------
    rng = _lcg(77)
    mism = 0
    tested = 0
    for _ in range(200):
        n = 1 + rng() % 15
        array = [(rng() % 30) - 15 for _ in range(n)]
        rk = RangeKth(array)
        for _ in range(8):
            a = rng() % n
            b = rng() % n
            l, r = min(a, b), max(a, b)
            length = r - l + 1
            k = 1 + rng() % length
            got = rk.kth_smallest(l, r, k)
            exp = brute_kth_smallest(array, l, r, k)
            tested += 1
            if got != exp:
                mism += 1
    check("range k-th smallest == brute sort (200 arrays)", mism == 0, f"{mism}/{tested}")

    # ---- 4. order statistics: min, median, max ------------------------------------------
    rng = _lcg(7)
    ok = True
    for _ in range(100):
        n = 1 + rng() % 12
        array = [rng() % 50 for _ in range(n)]
        rk = RangeKth(array)
        a = rng() % n
        b = rng() % n
        l, r = min(a, b), max(a, b)
        length = r - l + 1
        sub = sorted(array[l:r + 1])
        if rk.kth_smallest(l, r, 1) != sub[0]:
            ok = False
        if rk.kth_smallest(l, r, length) != sub[-1]:
            ok = False
        med_k = (length + 1) // 2
        if rk.kth_smallest(l, r, med_k) != sub[med_k - 1]:
            ok = False
    check("min / median / max order statistics correct", ok)

    # ---- 5. range_rank (count <= value) -------------------------------------------------
    rng = _lcg(321)
    mism = 0
    for _ in range(100):
        n = 1 + rng() % 12
        array = [(rng() % 20) for _ in range(n)]
        rk = RangeKth(array)
        a = rng() % n
        b = rng() % n
        l, r = min(a, b), max(a, b)
        val = rng() % 20
        got = rk.range_rank(l, r, val)
        exp = sum(1 for x in array[l:r + 1] if x <= val)
        if got != exp:
            mism += 1
    check("range_rank (count <= value) matches brute", mism == 0, f"{mism}")

    # ---- 6. hand example ----------------------------------------------------------------
    array = [5, 2, 8, 1, 9, 3, 7]
    rk = RangeKth(array)
    check("kth: 1st smallest in [0,6] = 1", rk.kth_smallest(0, 6, 1) == 1)
    check("kth: 4th smallest in [0,6] = 5", rk.kth_smallest(0, 6, 4) == 5)
    check("kth: 2nd smallest in [1,3] (2,8,1) = 2", rk.kth_smallest(1, 3, 2) == 2)
    check("kth: max in [2,4] (8,1,9) = 9", rk.kth_smallest(2, 4, 3) == 9)

    # ---- 7. edge cases ------------------------------------------------------------------
    single = RangeKth([42])
    check("single-element array kth", single.kth_smallest(0, 0, 1) == 42)
    check("duplicate values", RangeKth([3, 3, 3]).kth_smallest(0, 2, 2) == 3)
    try:
        PersistentSegmentTree(0)
        check("size=0 raises", False)
    except ValueError:
        check("size=0 raises", True)
    try:
        rk.kth_smallest(0, 2, 10)
        check("k out of range raises", False)
    except ValueError:
        check("k out of range raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

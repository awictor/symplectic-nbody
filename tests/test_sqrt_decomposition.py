"""Tests for square-root decomposition: query/update vs brute force, block invariants, edge ranges."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sqrt_decomposition as SD  # noqa: E402


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
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main():
    # ---- 1. static queries match brute force for all ranges, all ops --------------------
    a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    for op in ("sum", "min", "max"):
        sd = SD.SqrtDecomposition(a, op)
        ok = all(sd.query(l, r) == SD.brute_query(a, l, r, op)
                 for l in range(len(a)) for r in range(l, len(a)))
        check(f"static {op} queries match brute for all ranges", ok)

    # ---- 2. block size is Theta(sqrt n) -------------------------------------------------
    for n in (16, 100, 1000):
        sd = SD.SqrtDecomposition(list(range(n)), "sum")
        check(f"n={n}: block size near sqrt(n)", abs(sd.block - math.isqrt(n)) <= 1, f"{sd.block}")

    # ---- 3. randomized interleaved updates + queries vs brute ---------------------------
    rnd = _lcg(7)
    for op in ("sum", "min", "max"):
        a = [int(rnd() * 100) for _ in range(120)]
        sd = SD.SqrtDecomposition(a, op)
        mism = 0
        for _ in range(800):
            if rnd() < 0.3:
                i = int(rnd() * 120)
                v = int(rnd() * 100)
                sd.update(i, v)
                a[i] = v
            else:
                l = int(rnd() * 120)
                r = int(rnd() * 120)
                if l > r:
                    l, r = r, l
                if sd.query(l, r) != SD.brute_query(a, l, r, op):
                    mism += 1
        check(f"randomized {op}: 800 ops match brute", mism == 0, f"{mism}")

    # ---- 4. block aggregates stay consistent after updates ------------------------------
    a = [int(rnd() * 50) for _ in range(60)]
    sd = SD.SqrtDecomposition(a, "sum")
    for _ in range(50):
        i = int(rnd() * 60)
        v = int(rnd() * 50)
        sd.update(i, v)
    # every block aggregate must equal the sum of its underlying elements
    ok = True
    arr = sd.to_list()
    for bi in range(sd.block_count()):
        lo = bi * sd.block
        hi = min(lo + sd.block, sd.n)
        if sd.blocks[bi] != sum(arr[lo:hi]):
            ok = False
    check("block aggregates consistent with array", ok)

    # ---- 5. single-element and full-array ranges ----------------------------------------
    sd = SD.SqrtDecomposition([5, 2, 8, 1, 9, 3], "sum")
    check("single-element range", sd.query(2, 2) == 8)
    check("full-array range sum", sd.query(0, 5) == 28)
    sdm = SD.SqrtDecomposition([5, 2, 8, 1, 9, 3], "min")
    check("full-array min", sdm.query(0, 5) == 1)
    sdx = SD.SqrtDecomposition([5, 2, 8, 1, 9, 3], "max")
    check("full-array max", sdx.query(0, 5) == 9)

    # ---- 6. empty range returns the identity --------------------------------------------
    sd = SD.SqrtDecomposition([1, 2, 3], "sum")
    check("empty range (l>r) sum is 0", sd.query(2, 1) == 0)
    check("empty range min is +inf", SD.SqrtDecomposition([1, 2, 3], "min").query(2, 1) == math.inf)

    # ---- 7. update then query reflects the change ---------------------------------------
    sd = SD.SqrtDecomposition([1, 1, 1, 1, 1, 1, 1, 1, 1], "sum")
    before = sd.query(0, 8)
    sd.update(4, 10)
    check("update changes the query result", sd.query(0, 8) == before + 9)
    check("update visible in a range containing it", sd.query(3, 5) == 12)
    check("update invisible to a range not containing it", sd.query(0, 3) == 4)

    # ---- 8. range assignment ------------------------------------------------------------
    sd = SD.SqrtDecomposition([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "sum")
    sd.range_assign(2, 6, 0)
    check("range_assign zeroes the span", sd.query(2, 6) == 0)
    check("range_assign leaves the rest", sd.query(0, 1) == 3 and sd.query(7, 9) == 27)
    # aggregates still consistent
    arr = sd.to_list()
    ok = all(sd.blocks[bi] == sum(arr[bi * sd.block:min((bi + 1) * sd.block, sd.n)])
             for bi in range(sd.block_count()))
    check("range_assign keeps block aggregates consistent", ok)

    # ---- 9. queries spanning block boundaries -------------------------------------------
    a = list(range(1, 26))                        # 1..25, block size 5
    sd = SD.SqrtDecomposition(a, "sum")
    check("cross-boundary query [3,17]", sd.query(3, 17) == sum(a[3:18]))
    check("cross-boundary query [7,7]", sd.query(7, 7) == a[7])

    # ---- 10. singleton array ------------------------------------------------------------
    sd = SD.SqrtDecomposition([42], "sum")
    check("singleton query", sd.query(0, 0) == 42)
    sd.update(0, 7)
    check("singleton update", sd.query(0, 0) == 7)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

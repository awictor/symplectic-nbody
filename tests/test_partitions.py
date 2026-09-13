"""Tests for integer partitions: pentagonal recurrence vs brute/DP, Euler's theorem, conjugate."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from partitions import (  # noqa: E402
    partition_count,
    partition_counts_up_to,
    generate_partitions,
    generate_distinct_partitions,
    count_distinct_partitions,
    count_odd_partitions,
    count_partitions_into_k_parts,
    conjugate,
    brute_partition_count,
    dp_partition_count,
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


def main():
    # ---- 1. pentagonal recurrence matches brute and DP counts ---------------------------
    mism_b = mism_d = 0
    for n in range(0, 26):
        pc = partition_count(n)
        if pc != brute_partition_count(n):
            mism_b += 1
        if pc != dp_partition_count(n):
            mism_d += 1
    check("pentagonal p(n) == brute enumeration (n<=25)", mism_b == 0, f"{mism_b}")
    check("pentagonal p(n) == knapsack DP (n<=25)", mism_d == 0, f"{mism_d}")

    # ---- 2. known values ----------------------------------------------------------------
    check("p(0) = 1", partition_count(0) == 1)
    check("p(1) = 1", partition_count(1) == 1)
    check("p(4) = 5", partition_count(4) == 5)
    check("p(10) = 42", partition_count(10) == 42)
    check("p(50) = 204226", partition_count(50) == 204226)
    check("p(100) = 190569292", partition_count(100) == 190569292)

    # ---- 3. counts_up_to consistent with individual calls -------------------------------
    up = partition_counts_up_to(30)
    check("counts_up_to matches individual", all(up[n] == partition_count(n) for n in range(31)))

    # ---- 4. every generated partition sums to n, all distinct ---------------------------
    ok_sum = ok_uniq = True
    for n in range(0, 16):
        parts = list(generate_partitions(n))
        if any(sum(p) != n for p in parts):
            ok_sum = False
        if len(set(parts)) != len(parts):
            ok_uniq = False
        # non-increasing
        if any(list(p) != sorted(p, reverse=True) for p in parts):
            ok_sum = False
    check("generated partitions sum to n and are non-increasing", ok_sum)
    check("generated partitions all distinct", ok_uniq)

    # ---- 5. generation count matches p(n) -----------------------------------------------
    check("generation count == p(n)", all(len(list(generate_partitions(n))) == partition_count(n)
                                          for n in range(0, 18)))

    # ---- 6. Euler's theorem: #distinct-part == #odd-part --------------------------------
    ok = True
    for n in range(0, 40):
        if count_distinct_partitions(n) != count_odd_partitions(n):
            ok = False
    check("Euler: distinct-part count == odd-part count (n<40)", ok)
    # and generation agrees with the DP count for distinct
    check("distinct generation count == DP count",
          all(len(list(generate_distinct_partitions(n))) == count_distinct_partitions(n)
              for n in range(0, 20)))

    # ---- 7. partitions into exactly k parts sum to p(n) ---------------------------------
    ok = True
    for n in range(1, 20):
        s = sum(count_partitions_into_k_parts(n, k) for k in range(1, n + 1))
        if s != partition_count(n):
            ok = False
    check("sum over k of p(n,k) == p(n)", ok)
    check("p(n,1) = 1 (the single part)", all(count_partitions_into_k_parts(n, 1) == 1
                                              for n in range(1, 10)))
    check("p(n,n) = 1 (all ones)", all(count_partitions_into_k_parts(n, n) == 1
                                       for n in range(1, 10)))

    # ---- 8. conjugate is an involution and swaps largest-part <-> num-parts -------------
    ok_inv = ok_swap = True
    for n in range(1, 14):
        for p in generate_partitions(n):
            c = conjugate(p)
            if conjugate(c) != p:
                ok_inv = False
            if sum(c) != n:
                ok_inv = False
            # largest part of p == number of parts of conjugate
            if p and c and c[0] != len(p):
                ok_swap = False
    check("conjugate is an involution and preserves sum", ok_inv)
    check("conjugate swaps largest-part and number-of-parts", ok_swap)

    # ---- 9. self-conjugate partitions == partitions into distinct odd parts -------------
    # a classic identity: #self-conjugate partitions of n == #partitions of n into distinct odd parts
    def count_self_conjugate(n):
        return sum(1 for p in generate_partitions(n) if conjugate(p) == p)

    def count_distinct_odd(n):
        c = 0
        for p in generate_distinct_partitions(n):
            if all(x % 2 == 1 for x in p):
                c += 1
        return c
    ok = all(count_self_conjugate(n) == count_distinct_odd(n) for n in range(1, 22))
    check("self-conjugate count == distinct-odd-parts count", ok)

    # ---- 10. bounded-part partitions ----------------------------------------------------
    # partitions of 6 with parts <= 3
    parts = list(generate_partitions(6, max_part=3))
    check("partitions of 6 into parts <=3 all obey the cap", all(max(p) <= 3 for p in parts if p))
    check("partitions of 6 into parts <=3 count", len(parts) == 7)  # 3+3,3+2+1,3+1+1+1,2+2+2,2+2+1+1,2+1^4,1^6

    # ---- 11. edge cases -----------------------------------------------------------------
    check("p(negative) = 0", partition_count(-3) == 0)
    check("empty partition of 0", list(generate_partitions(0)) == [()])
    check("p(n,0) = 0 for n>0", count_partitions_into_k_parts(5, 0) == 0)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

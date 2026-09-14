"""Tests for Egyptian fractions: greedy sums back, distinct increasing, Engel, Sylvester sequence."""

import os
import sys
from fractions import Fraction
from math import gcd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from egyptian_fraction import (  # noqa: E402
    greedy_egyptian, egyptian_sum, engel_expansion, engel_to_fraction,
    sylvester_sequence, sylvester_reciprocal_sum, all_distinct, is_increasing,
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
    # ---- 1. greedy decomposition sums back to the input for many fractions --------------
    ok = True
    for den in range(2, 60):
        for num in range(1, den):
            if gcd(num, den) != 1:
                continue
            denoms = greedy_egyptian(num, den)
            if egyptian_sum(denoms) != Fraction(num, den):
                ok = False
                check("greedy sums back", False, f"{num}/{den}: {denoms}")
                break
        if not ok:
            break
    if ok:
        check("greedy Egyptian sums back to input (all reduced n/d, d<60)", True)

    # ---- 2. denominators are distinct and strictly increasing ---------------------------
    ok = True
    for den in range(2, 60):
        for num in range(1, den):
            if gcd(num, den) != 1:
                continue
            denoms = greedy_egyptian(num, den)
            if not (all_distinct(denoms) and is_increasing(denoms)):
                ok = False
                break
        if not ok:
            break
    check("greedy denominators distinct + strictly increasing", ok)

    # ---- 3. known small cases -----------------------------------------------------------
    check("2/3 = 1/2 + 1/6", greedy_egyptian(2, 3) == [2, 6])
    check("3/7 = 1/3 + 1/11 + 1/231", greedy_egyptian(3, 7) == [3, 11, 231])
    check("1/2 = 1/2", greedy_egyptian(1, 2) == [2])
    check("5/6 = 1/2 + 1/3", greedy_egyptian(5, 6) == [2, 3])

    # ---- 4. numerator strictly decreases each greedy step -------------------------------
    # (Fibonacci's termination proof)
    ok = True
    for num, den in [(5, 121), (8, 97), (31, 311)]:
        x = Fraction(num, den)
        prev_num = x.numerator
        while x > 0:
            d = (x.denominator + x.numerator - 1) // x.numerator
            x -= Fraction(1, d)
            if x > 0 and x.numerator >= prev_num:
                ok = False
                break
            if x > 0:
                prev_num = x.numerator
        if not ok:
            break
    check("greedy numerator strictly decreases (termination)", ok)

    # ---- 5. rejects values outside (0,1) ------------------------------------------------
    for bad in [(0, 1), (1, 1), (3, 2)]:
        try:
            greedy_egyptian(*bad)
            check(f"reject {bad}", False)
        except ValueError:
            check(f"reject {bad[0]}/{bad[1]} outside (0,1)", True)

    # ---- 6. Engel expansion sums back ---------------------------------------------------
    ok = True
    for den in range(2, 40):
        for num in range(1, den):
            if gcd(num, den) != 1:
                continue
            a = engel_expansion(num, den)
            if engel_to_fraction(a) != Fraction(num, den):
                ok = False
                check("Engel sums back", False, f"{num}/{den}: {a}")
                break
        if not ok:
            break
    if ok:
        check("Engel expansion reconstructs the fraction (d<40)", True)

    # ---- 7. Engel expansion is non-decreasing -------------------------------------------
    ok = True
    for num, den in [(3, 7), (5, 12), (7, 15), (11, 30)]:
        a = engel_expansion(num, den)
        if any(a[i] > a[i + 1] for i in range(len(a) - 1)):
            ok = False
            break
    check("Engel expansion non-decreasing", ok)

    # ---- 8. Engel of 1/2 and known values -----------------------------------------------
    check("Engel(3/7) = [3,4,7]", engel_expansion(3, 7) == [3, 4, 7])

    # ---- 9. Sylvester's sequence recurrence ---------------------------------------------
    seq = sylvester_sequence(6)
    check("Sylvester = 2,3,7,43,1807,3263443", seq == [2, 3, 7, 43, 1807, 3263443], f"{seq}")
    # each term = 1 + product of all previous
    ok = True
    prod = 1
    for term in seq:
        if term != prod + 1:
            ok = False
            break
        prod *= term
    check("Sylvester recurrence a_k = 1 + prod(prev)", ok)

    # ---- 10. Sylvester reciprocals approach 1 from below --------------------------------
    for n in [1, 2, 3, 4, 5]:
        s = sylvester_reciprocal_sum(n)
        check(f"Sylvester reciprocal sum({n}) < 1", s < 1)
    # and the gap to 1 is exactly 1/(prod - 1)... check it shrinks
    s5 = sylvester_reciprocal_sum(5)
    s4 = sylvester_reciprocal_sum(4)
    check("Sylvester reciprocal sum increases toward 1", s4 < s5 < 1)

    # ---- 11. greedy expansion of a value near 1 is short (Sylvester-like) ---------------
    # 1 - 1/big has a short-ish greedy expansion; just confirm it sums back
    denoms = greedy_egyptian(1806, 1807)
    check("greedy near-1 sums back", egyptian_sum(denoms) == Fraction(1806, 1807))

    # ---- 12. all-distinct / increasing helpers ------------------------------------------
    check("all_distinct true", all_distinct([2, 6, 42]))
    check("all_distinct false", not all_distinct([2, 2, 6]))
    check("is_increasing true", is_increasing([2, 6, 42]))
    check("is_increasing false", not is_increasing([6, 2]))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for Pell: known fundamental solutions, x^2-Dy^2=1 exact, recurrence, negative Pell, CF period."""

import os
import sys
from math import isqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pell import (  # noqa: E402
    cf_sqrt_period,
    fundamental_solution,
    negative_pell_solution,
    solutions,
    verify,
    is_square,
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
    # ---- 1. known fundamental solutions -------------------------------------------------
    known = {
        2: (3, 2), 3: (2, 1), 5: (9, 4), 6: (5, 2), 7: (8, 3), 13: (649, 180),
        61: (1766319049, 226153980),          # Fermat's challenge
        109: (158070671986249, 15140424455100),
    }
    ok = True
    for D, expected in known.items():
        if fundamental_solution(D) != expected:
            ok = False
            check("known fundamental solution", False, f"D={D}: {fundamental_solution(D)} vs {expected}")
            break
    if ok:
        check("fundamental solutions match known values (incl D=61, 109)", True)

    # ---- 2. fundamental solution satisfies x^2 - D y^2 = 1 for all non-square D ---------
    ok = True
    for D in range(2, 200):
        if is_square(D):
            continue
        x, y = fundamental_solution(D)
        if not verify(D, x, y, 1):
            ok = False
            check("fundamental satisfies Pell", False, f"D={D}")
            break
    if ok:
        check("fundamental solution solves x^2-Dy^2=1 (D=2..199)", True)

    # ---- 3. perfect squares have no solution --------------------------------------------
    check("D=4 (square) -> None", fundamental_solution(4) is None)
    check("D=9 (square) -> None", fundamental_solution(9) is None)
    check("cf period empty for squares", cf_sqrt_period(16) == (4, []))

    # ---- 4. recurrence generates further valid solutions --------------------------------
    ok = True
    for D in [2, 3, 5, 7, 13, 61]:
        sols = solutions(D, 5)
        for x, y in sols:
            if not verify(D, x, y, 1):
                ok = False
                break
        # solutions strictly increasing
        if [s[0] for s in sols] != sorted(s[0] for s in sols) or len(set(sols)) != len(sols):
            ok = False
        if not ok:
            check("recurrence solutions valid", False, f"D={D}: {sols}")
            break
    if ok:
        check("recurrence generates valid increasing solutions (6 D)", True)

    # ---- 5. second solution = square of the fundamental (Brahmagupta composition) -------
    D = 2
    x1, y1 = fundamental_solution(D)
    sols = solutions(D, 2)
    # (x1 + y1 sqrt2)^2 = (x1^2 + 2 y1^2) + (2 x1 y1) sqrt2
    x2_expected = x1 * x1 + D * y1 * y1
    y2_expected = 2 * x1 * y1
    check("2nd solution = fundamental squared", sols[1] == (x2_expected, y2_expected),
          f"{sols[1]} vs ({x2_expected},{y2_expected})")

    # ---- 6. negative Pell exists iff CF period is odd -----------------------------------
    # D=2: period [2] length 1 (odd) -> neg Pell solvable (1^2 - 2*1^2 = -1)
    neg = negative_pell_solution(2)
    check("negative Pell D=2 = (1,1)", neg == (1, 1) and verify(2, neg[0], neg[1], -1), f"{neg}")
    # D=5: sqrt5 = [2; 4,...] period [4]? actually [2;4] len 1 odd -> (2,1): 4-5=-1
    neg5 = negative_pell_solution(5)
    check("negative Pell D=5", neg5 is not None and verify(5, neg5[0], neg5[1], -1), f"{neg5}")
    # D=3: period even -> no negative solution
    check("negative Pell D=3 -> None", negative_pell_solution(3) is None)
    # D=7: period even -> none
    check("negative Pell D=7 -> None", negative_pell_solution(7) is None)

    # ---- 7. negative Pell existence matches CF-period parity across a range -------------
    ok = True
    for D in range(2, 100):
        if is_square(D):
            continue
        _, period = cf_sqrt_period(D)
        odd = len(period) % 2 == 1
        has_neg = negative_pell_solution(D) is not None
        if odd != has_neg:
            ok = False
            check("neg Pell iff odd period", False, f"D={D}: odd={odd} has_neg={has_neg}")
            break
        if has_neg:
            x, y = negative_pell_solution(D)
            if not verify(D, x, y, -1):
                ok = False
                check("neg Pell verifies", False, f"D={D}")
                break
    if ok:
        check("negative Pell exists iff CF period odd (D=2..99)", True)

    # ---- 8. cf period of sqrt(2) is [2], sqrt(3) is [1,2] -------------------------------
    check("cf sqrt(2) = (1, [2])", cf_sqrt_period(2) == (1, [2]))
    check("cf sqrt(3) = (1, [1,2])", cf_sqrt_period(3) == (1, [1, 2]))
    check("cf sqrt(7) = (2, [1,1,1,4])", cf_sqrt_period(7) == (2, [1, 1, 1, 4]))

    # ---- 9. cf period is symmetric palindrome + final term 2a0 -------------------------
    # the period (minus its last term) is a palindrome; last term is 2*a0
    ok = True
    for D in [2, 3, 5, 7, 13, 23, 31]:
        a0, period = cf_sqrt_period(D)
        if period[-1] != 2 * a0:
            ok = False
            break
        palindrome = period[:-1]
        if palindrome != palindrome[::-1]:
            ok = False
            break
    check("cf period palindrome + last=2a0", ok)

    # ---- 10. solutions correspond to powers in Z[sqrt D] --------------------------------
    D = 3
    x1, y1 = fundamental_solution(D)
    sols = solutions(D, 4)
    # verify (x_n + y_n sqrt3) = (x1 + y1 sqrt3)^n by recomputing
    px, py = x1, y1
    ok = sols[0] == (x1, y1)
    for n in range(1, 4):
        px, py = x1 * px + D * y1 * py, x1 * py + y1 * px
        if sols[n] != (px, py):
            ok = False
    check("solutions are powers in Z[sqrt D]", ok)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

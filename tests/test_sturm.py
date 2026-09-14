"""Tests for Sturm's theorem: root counts match Durand-Kerner, isolation, refinement, repeated roots."""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sturm import (  # noqa: E402
    count_real_roots, count_roots_in, isolate_roots, refine_root,
    sturm_sequence, make_squarefree, _eval,
)
from durand_kerner import roots as dk_roots, from_roots  # noqa: E402


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


def _distinct_real_roots(coeffs, tol=1e-6):
    """Count distinct real roots via Durand-Kerner (reference)."""
    rs = dk_roots(coeffs)
    reals = [r.real for r in rs if abs(r.imag) < tol]
    reals.sort()
    distinct = []
    for r in reals:
        if not distinct or abs(r - distinct[-1]) > tol:
            distinct.append(r)
    return len(distinct)


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    # ---- 1. known cases -----------------------------------------------------------------
    check("x^2 - 2 has 2 real roots", count_real_roots([1, 0, -2]) == 2)
    check("(x-1)(x-2)(x-3) has 3", count_real_roots([1, -6, 11, -6]) == 3)
    check("x^2 + 1 has 0", count_real_roots([1, 0, 1]) == 0)
    check("x^3 - x has 3 (-1,0,1)", count_real_roots([1, 0, -1, 0]) == 3)
    check("x^4 + 1 has 0", count_real_roots([1, 0, 0, 0, 1]) == 0)

    # ---- 2. total count matches Durand-Kerner distinct-real count -----------------------
    rng = _lcg(1)
    ok = True
    for _ in range(30):
        deg = 2 + int(rng() * 4)
        coeffs = [1] + [int(rng() * 11) - 5 for _ in range(deg)]
        sturm_n = count_real_roots(coeffs)
        dk_n = _distinct_real_roots(coeffs)
        if sturm_n != dk_n:
            ok = False
            check("Sturm == Durand-Kerner", False, f"{coeffs}: sturm={sturm_n} dk={dk_n}")
            break
    if ok:
        check("Sturm real-root count == Durand-Kerner distinct reals (30 polys)", True)

    # ---- 3. built-from-roots polynomials count correctly --------------------------------
    ok = True
    for real_roots_list in [[1, 2, 3], [-2, -1, 0, 5], [1.5, 1.5], [0, 0, 0]]:
        coeffs = [c.real for c in from_roots([complex(r) for r in real_roots_list])]
        # distinct real roots
        distinct = len(set(real_roots_list))
        if count_real_roots(coeffs) != distinct:
            ok = False
            check("from_roots count", False, f"{real_roots_list}: {count_real_roots(coeffs)} vs {distinct}")
            break
    if ok:
        check("count == distinct roots for from_roots polynomials", True)

    # ---- 4. interval counts -------------------------------------------------------------
    # (x-1)(x-2)(x-3): (0,1.5] has 1, (1.5,2.5] has 1, (0,4] has 3
    p = [1, -6, 11, -6]
    check("(0,1.5] has 1 root", count_roots_in(p, 0, Fraction(3, 2)) == 1)
    check("(1.5,2.5] has 1 root", count_roots_in(p, Fraction(3, 2), Fraction(5, 2)) == 1)
    check("(0,4] has 3 roots", count_roots_in(p, 0, 4) == 3)
    check("(4,10] has 0 roots", count_roots_in(p, 4, 10) == 0)

    # ---- 5. isolate_roots gives one root per interval, right count ----------------------
    intervals = isolate_roots(p)
    check("3 isolating intervals", len(intervals) == 3, f"{len(intervals)}")
    ok = all(count_roots_in(p, a, b) == 1 for a, b in intervals)
    check("each interval has exactly one root", ok)
    # each interval brackets one of 1, 2, 3
    for target in [1, 2, 3]:
        inside = any(a < target <= b for a, b in intervals)
        check(f"root {target} isolated", inside)

    # ---- 6. refine_root converges to the true root --------------------------------------
    # sqrt(2) in (1,2)
    lo, hi = refine_root([1, 0, -2], (1, 2), width=Fraction(1, 10 ** 12))
    approx = float((lo + hi) / 2)
    check("refined sqrt(2)", abs(approx - 2 ** 0.5) < 1e-9, f"{approx}")

    # ---- 7. repeated roots counted once (distinct) --------------------------------------
    # (x-1)^2 (x-2) = x^3 -4x^2 +5x -2 has distinct roots {1, 2}
    check("(x-1)^2(x-2) has 2 distinct roots", count_real_roots([1, -4, 5, -2]) == 2)
    # (x-1)^3 has 1 distinct root
    check("(x-1)^3 has 1 distinct root", count_real_roots([1, -3, 3, -1]) == 1)

    # ---- 8. make_squarefree removes multiplicity ----------------------------------------
    sq = make_squarefree([1, -3, 3, -1])  # (x-1)^3 -> (x-1)
    # squarefree of (x-1)^3 is proportional to (x-1): degree 1
    check("squarefree of (x-1)^3 is linear", len([c for c in sq]) == 2, f"deg {len(sq)-1}")

    # ---- 9. Sturm sequence starts with p and p' -----------------------------------------
    seq = sturm_sequence([1, 0, -2])  # p = x^2-2, p' = 2x
    check("Sturm seq[0] = p", [float(c) for c in seq[0]] == [1, 0, -2])
    check("Sturm seq[1] = p'", [float(c) for c in seq[1]] == [2, 0])

    # ---- 10. linear and constant polynomials --------------------------------------------
    check("x - 5 has 1 root", count_real_roots([1, -5]) == 1)
    check("constant 3 has 0 roots", count_real_roots([3]) == 0)

    # ---- 11. high-degree with all real roots --------------------------------------------
    # (x)(x-1)(x-2)(x-3)(x-4) has 5 distinct roots
    coeffs = [c.real for c in from_roots([0j, 1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j])]
    check("degree-5 all-real has 5 roots", count_real_roots(coeffs) == 5)

    # ---- 12. no real roots for even-degree positive-definite ----------------------------
    check("x^4 + x^2 + 1 has 0 real roots", count_real_roots([1, 0, 1, 0, 1]) == 0)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

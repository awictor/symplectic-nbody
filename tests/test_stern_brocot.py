"""Tests for Stern-Brocot / Farey: path round-trips, CF match, best approx, Farey unimodular relation."""

import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stern_brocot import (  # noqa: E402
    stern_brocot_path,
    from_path,
    continued_fraction_of,
    best_rational_approximation,
    farey_sequence,
    farey_neighbours_ok,
    brute_farey,
    brute_best_approximation,
    mediant,
)
from continued_fraction import best_approximation as cf_best  # noqa: E402


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
    # ---- 1. every rational's path reconstructs it exactly -------------------------------
    ok = True
    for p in range(1, 25):
        for q in range(1, 25):
            g = math.gcd(p, q)
            rp, rq = p // g, q // g
            path = stern_brocot_path(rp, rq)
            if from_path(path) != (rp, rq):
                ok = False
                check("path round-trip", False, f"{rp}/{rq} -> {path} -> {from_path(path)}")
                break
        if not ok:
            break
    if ok:
        check("Stern-Brocot path reconstructs every rational (p,q<=24)", True)

    # ---- 2. root is 1/1 with empty path -------------------------------------------------
    check("1/1 has empty path", stern_brocot_path(1, 1) == "")
    check("from_path('') = 1/1", from_path("") == (1, 1))

    # ---- 3. path directions: 1/2 is Left, 2/1 is Right ----------------------------------
    check("1/2 path is 'L'", stern_brocot_path(1, 2) == "L")
    check("2/1 path is 'R'", stern_brocot_path(2, 1) == "R")

    # ---- 4. continued fraction of p/q matches gcd expansion -----------------------------
    check("CF of 355/113", continued_fraction_of(355, 113) == [3, 7, 16],
          f"{continued_fraction_of(355, 113)}")
    check("CF of 7/1", continued_fraction_of(7, 1) == [7])

    # ---- 5. best approximation agrees with continued-fraction best_approximation --------
    import math as m
    ok = True
    for x, N in [(m.pi, 100), (m.pi, 1000), (m.e, 50), (m.sqrt(2), 100), (1.5, 10), (0.1, 7)]:
        sb = best_rational_approximation(x, N)
        cf = cf_best(x, N)
        cf_frac = Fraction(cf[0], cf[1]) if isinstance(cf, tuple) else Fraction(cf)
        if abs(float(sb) - x) > abs(float(cf_frac) - x) + 1e-12:
            ok = False
            check("SB best == CF best", False, f"x={x} N={N}: {sb} vs {cf_frac}")
            break
    if ok:
        check("best approximation matches continued-fraction best (6 cases)", True)

    # ---- 6. best approximation beats every fraction of no-larger denominator (brute) ----
    ok = True
    for x, N in [(math.pi, 50), (math.e, 40), (math.sqrt(2), 60), (0.618, 30)]:
        sb = best_rational_approximation(x, N)
        brute = brute_best_approximation(x, N)
        if abs(float(sb) - x) > abs(float(brute) - x) + 1e-12:
            ok = False
            check("SB best <= brute best", False, f"x={x} N={N}: {sb} vs {brute}")
            break
    if ok:
        check("best approximation is optimal (brute force, 4 cases)", True)

    # ---- 7. pi ~ 22/7 at small denominator, 355/113 at larger --------------------------
    check("pi ~ 22/7 (N=10)", best_rational_approximation(math.pi, 10) == Fraction(22, 7),
          f"{best_rational_approximation(math.pi, 10)}")
    check("pi ~ 355/113 (N=200)", best_rational_approximation(math.pi, 200) == Fraction(355, 113),
          f"{best_rational_approximation(math.pi, 200)}")

    # ---- 8. Farey sequence matches brute enumeration ------------------------------------
    ok = True
    for n in range(1, 12):
        if farey_sequence(n) != brute_farey(n):
            ok = False
            check("Farey == brute", False, f"n={n}")
            break
    if ok:
        check("Farey sequence == brute enumeration (n=1..11)", True)

    # ---- 9. Farey neighbours satisfy the unimodular relation bc - ad = 1 ----------------
    ok = all(farey_neighbours_ok(farey_sequence(n)) for n in range(1, 15))
    check("Farey neighbours: bc - ad = 1", ok)

    # ---- 10. Farey F_5 is the known sequence --------------------------------------------
    f5 = [str(f) for f in farey_sequence(5)]
    expected = ["0", "1/5", "1/4", "1/3", "2/5", "1/2", "3/5", "2/3", "3/4", "4/5", "1"]
    check("F_5 correct", f5 == expected, f"{f5}")

    # ---- 11. mediant of Farey neighbours appears between them at the right level --------
    # in F_2 = [0, 1/2, 1], mediant of 0/1 and 1/2 is 1/3, which appears in F_3
    f3 = farey_sequence(3)
    check("mediant of 0 and 1/2 is 1/3 in F_3", Fraction(1, 3) in f3)

    # ---- 12. F_n length grows and F_1 = [0, 1] ------------------------------------------
    check("F_1 = [0, 1]", farey_sequence(1) == [Fraction(0), Fraction(1)])
    check("Farey lengths increasing", all(len(farey_sequence(n)) < len(farey_sequence(n + 1))
                                          for n in range(1, 10)))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

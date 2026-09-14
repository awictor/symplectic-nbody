"""Tests for resultant/discriminant: common-root detection, b^2-4ac, root-difference product, elimination."""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from resultant import (  # noqa: E402
    resultant, discriminant, has_common_root, has_repeated_root,
    sylvester_matrix, derivative, product_of_p_over_roots,
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


def main():
    # ---- 1. resultant is zero exactly when polynomials share a root ---------------------
    # (x-1)(x-2) and (x-2)(x-3): share 2
    check("shared root -> resultant 0", resultant([1, -3, 2], [1, -5, 6]) == 0)
    # (x-1)(x-2) and (x-3)(x-4): no shared root
    check("no shared root -> resultant != 0", resultant([1, -3, 2], [1, -7, 12]) != 0)
    check("has_common_root true", has_common_root([1, -3, 2], [1, -5, 6]))
    check("has_common_root false", not has_common_root([1, -3, 2], [1, -7, 12]))

    # ---- 2. discriminant of a quadratic is b^2 - 4ac ------------------------------------
    ok = True
    for a, b, c in [(1, -5, 6), (2, 3, 1), (1, 0, -2), (3, -1, -4)]:
        disc = discriminant([a, b, c])
        if disc != b * b - 4 * a * c:
            ok = False
            check("quadratic discriminant", False, f"{a,b,c}: {disc} vs {b*b-4*a*c}")
            break
    if ok:
        check("quadratic discriminant = b^2 - 4ac", True)

    # ---- 3. discriminant is zero exactly for repeated roots -----------------------------
    check("(x-1)^2 has repeated root", has_repeated_root([1, -2, 1]))
    check("(x-1)(x-2) no repeated root", not has_repeated_root([1, -3, 2]))
    check("(x-1)^2(x-2) repeated", has_repeated_root([c.real for c in from_roots([1+0j, 1+0j, 2+0j])]))
    check("(x-1)(x-2)(x-3) not repeated",
          not has_repeated_root([c.real for c in from_roots([1+0j, 2+0j, 3+0j])]))

    # ---- 4. discriminant sign counts real roots of a quadratic --------------------------
    check("disc > 0 -> two real roots", discriminant([1, -5, 6]) > 0)   # roots 2,3
    check("disc < 0 -> complex pair", discriminant([1, 0, 1]) < 0)      # x^2+1
    check("disc = 0 -> double root", discriminant([1, -2, 1]) == 0)

    # ---- 5. cubic discriminant matches the classical formula ---------------------------
    # disc(x^3+px+q) = -4p^3 - 27q^2
    for p, q in [(-1, 0), (2, 3), (-3, 1)]:
        disc = discriminant([1, 0, p, q])
        check(f"cubic disc x^3+{p}x+{q} = -4p^3-27q^2",
              disc == -4 * p ** 3 - 27 * q ** 2, f"{disc} vs {-4*p**3-27*q**2}")

    # ---- 6. resultant = a_p^{deg q} * prod p(root of q) (up to the known identity) ------
    # Res(p,q) = a_p^{deg q} * prod over roots b of q of p(b)
    p = [1, -3, 2]      # (x-1)(x-2)
    q = [1, -7, 12]     # (x-3)(x-4)
    R = resultant(p, q)
    q_roots = dk_roots(q)
    prod = product_of_p_over_roots(p, q_roots)
    # a_p = 1, deg q = 2
    check("Res = a_p^deg(q) * prod p(q-roots)", abs(complex(R) - prod) < 1e-6,
          f"{R} vs {prod}")

    # ---- 7. resultant symmetry Res(p,q) = (-1)^{deg p deg q} Res(q,p) -------------------
    p = [1, -3, 2]
    q = [1, -7, 12]
    Rpq = resultant(p, q)
    Rqp = resultant(q, p)
    dp, dq = 2, 2
    check("resultant (anti)symmetry", Rpq == (-1) ** (dp * dq) * Rqp, f"{Rpq} {Rqp}")

    # ---- 8. Sylvester matrix has the right size -----------------------------------------
    M = sylvester_matrix([1, 2, 3], [1, 4])  # deg 2 + deg 1 = 3
    check("Sylvester matrix is 3x3", len(M) == 3 and all(len(r) == 3 for r in M))

    # ---- 9. resultant with a constant --------------------------------------------------
    # Res(c, q) = c^deg q
    check("Res(5, x^2-1) = 25", resultant([5], [1, 0, -1]) == 25)
    check("Res(x^2-1, 3) = 9", resultant([1, 0, -1], [3]) == 9)

    # ---- 10. elimination: solve a 2-variable system ------------------------------------
    # System: x^2 + y^2 = 1 (circle), y = x (line). Substitute -> resultant in x eliminates y.
    # As polynomials in y: p = y^2 + (x^2 - 1) -> coeffs [1, 0, x^2-1]; q = y - x -> [1, -x].
    # Res_y(p, q) = p(x) at y=x = x^2 + x^2 - 1 = 2x^2 - 1. Roots x = +/- 1/sqrt(2).
    # verify by plugging a symbolic x value: check the resultant polynomial's value matches
    from fractions import Fraction as F
    def res_at(xval):
        p = [F(1), F(0), F(xval) ** 2 - 1]
        q = [F(1), -F(xval)]
        return resultant(p, q)
    # 2x^2 - 1 at x = 1/sqrt2 is 0; test a rational point: at x=1, 2-1=1
    check("elimination resultant at x=1 is 1", res_at(1) == 1, f"{res_at(1)}")
    check("elimination resultant at x=0 is -1", res_at(0) == -1, f"{res_at(0)}")
    # confirms Res_y = 2x^2 - 1 (value 1 at x=1, -1 at x=0)

    # ---- 11. random polynomials sharing a planted root --------------------------------
    def _lcg(seed):
        st = seed & 0xFFFFFFFF
        def n():
            nonlocal st
            st = (1664525 * st + 1013904223) & 0xFFFFFFFF
            return st
        return n
    rng = _lcg(1)
    ok = True
    for _ in range(15):
        shared = (rng() % 11) - 5
        # p = (x - shared)(x - a), q = (x - shared)(x - b)
        a = (rng() % 11) - 5
        b = (rng() % 11) - 5
        p = [c.real for c in from_roots([complex(shared), complex(a)])]
        q = [c.real for c in from_roots([complex(shared), complex(b)])]
        if not has_common_root([round(c) for c in p], [round(c) for c in q]):
            ok = False
            break
    check("planted shared root always detected", ok)

    # ---- 12. discriminant of a linear polynomial ----------------------------------------
    # disc of ax+b: degree 1, Res(p, p') where p' = a constant -> a; disc = a^? ; just check no crash
    d = discriminant([2, -3])
    check("linear discriminant computes", isinstance(d, Fraction))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

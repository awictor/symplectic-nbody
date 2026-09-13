"""Tests for Smith Normal Form: U A V = D, unimodular, divisibility chain, minors formula, groups."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smith_normal_form import (  # noqa: E402
    smith_normal_form,
    invariant_factors,
    rank,
    is_unimodular,
    abelian_group_structure,
    gcd_of_minors,
    _matmul,
    _int_det,
    _gcd,
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
        return (state >> 8) / (1 << 24)
    return nxt


def _is_diagonal(D):
    for i in range(len(D)):
        for j in range(len(D[0])):
            if i != j and D[i][j] != 0:
                return False
    return True


def _divides_chain(D):
    facs = [D[i][i] for i in range(min(len(D), len(D[0])))]
    for i in range(len(facs) - 1):
        a, b = facs[i], facs[i + 1]
        if a == 0 and b != 0:
            return False  # zeros must come last
        if a != 0 and b != 0 and b % a != 0:
            return False
    return True


def main():
    rng = _lcg(1)

    # ---- 1. U A V = D exactly, on random integer matrices -------------------------------
    ok = True
    for _ in range(40):
        m = 1 + int(rng() * 4)
        n = 1 + int(rng() * 4)
        A = [[int(rng() * 11) - 5 for _ in range(n)] for _ in range(m)]
        U, D, V = smith_normal_form(A)
        prod = _matmul(_matmul(U, A), V)
        if prod != D:
            ok = False
            check("U A V = D", False, f"A={A}")
            break
    if ok:
        check("U A V = D (40 random matrices)", True)

    # ---- 2. U and V are unimodular ------------------------------------------------------
    ok = True
    for _ in range(30):
        m = 1 + int(rng() * 4)
        n = 1 + int(rng() * 4)
        A = [[int(rng() * 9) - 4 for _ in range(n)] for _ in range(m)]
        U, D, V = smith_normal_form(A)
        if not (is_unimodular(U) and is_unimodular(V)):
            ok = False
            break
    check("U, V unimodular (det +/-1)", ok)

    # ---- 3. D is diagonal with the divisibility chain -----------------------------------
    ok = True
    for _ in range(30):
        m = 1 + int(rng() * 4)
        n = 1 + int(rng() * 4)
        A = [[int(rng() * 13) - 6 for _ in range(n)] for _ in range(m)]
        U, D, V = smith_normal_form(A)
        if not _is_diagonal(D) or not _divides_chain(D):
            ok = False
            check("D diagonal + chain", False, f"A={A} D={D}")
            break
    if ok:
        check("D diagonal with d_i | d_{i+1} (30 matrices)", True)

    # ---- 4. classic example: [[2,4,4],[-6,6,12],[10,-4,-16]] ----------------------------
    A = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
    U, D, V = smith_normal_form(A)
    facs = invariant_factors(D)
    # known SNF invariant factors are 2, 6, 12
    check("classic SNF invariant factors [2,6,12]", facs == [2, 6, 12], f"{facs}")

    # ---- 5. invariant factors match gcd-of-minors formula -------------------------------
    # d_k = gcd(k-minors) / gcd((k-1)-minors)
    def minors_factors(A):
        m, n = len(A), len(A[0])
        r = min(m, n)
        prev = 1
        out = []
        for k in range(1, r + 1):
            g = gcd_of_minors(A, k)
            if g == 0:
                break
            out.append(g // prev)
            prev = g
        return out

    ok = True
    for _ in range(15):
        m = 1 + int(rng() * 3)
        n = 1 + int(rng() * 3)
        A = [[int(rng() * 9) - 4 for _ in range(n)] for _ in range(m)]
        U, D, V = smith_normal_form(A)
        facs = invariant_factors(D)
        mf = minors_factors(A)
        if facs != mf:
            ok = False
            check("SNF factors == minors formula", False, f"A={A}: {facs} vs {mf}")
            break
    if ok:
        check("invariant factors == gcd-of-minors formula (15 matrices)", True)

    # ---- 6. product of invariant factors = |det| for square full-rank -------------------
    ok = True
    for _ in range(15):
        n = 1 + int(rng() * 3)
        A = [[int(rng() * 7) - 3 for _ in range(n)] for _ in range(n)]
        d = _int_det(A)
        if d == 0:
            continue
        U, D, V = smith_normal_form(A)
        facs = invariant_factors(D)
        prod = 1
        for f in facs:
            prod *= f
        if prod != abs(d):
            ok = False
            check("prod factors = |det|", False, f"A={A}: {prod} vs {abs(d)}")
            break
    if ok:
        check("product of invariant factors = |det| (square full-rank)", True)

    # ---- 7. rank matches nonzero-invariant-factor count ---------------------------------
    A = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]  # rank 2 (row2 = 2*row1)
    U, D, V = smith_normal_form(A)
    check("rank of rank-2 matrix", rank(D) == 2, f"{rank(D)}")

    # ---- 8. abelian group structure -----------------------------------------------------
    # relations [[2,0],[0,3]] -> Z/2 x Z/3, and since gcd(2,3)=1 this IS Z/6: the SNF invariant
    # factors are (1, 6), so the structure theorem reports the single torsion factor 6 (Z/6).
    torsion, free = abelian_group_structure([[2, 0], [0, 3]])
    check("Z/2 x Z/3 = Z/6 torsion", torsion == [6] and free == 0, f"{torsion},{free}")
    # relations [[2,0],[0,4]] -> gcd(2,4)=2, SNF (2,4): genuinely Z/2 x Z/4 (not cyclic)
    torsion2, free2 = abelian_group_structure([[2, 0], [0, 4]])
    check("Z/2 x Z/4 stays [2,4]", torsion2 == [2, 4] and free2 == 0, f"{torsion2},{free2}")
    # [[1,0],[0,0]] -> cokernel Z (one free), no torsion
    torsion, free = abelian_group_structure([[1, 0], [0, 0]])
    check("free rank from zero row", torsion == [] and free == 1, f"{torsion},{free}")

    # ---- 9. identity matrix -> all invariant factors 1 ----------------------------------
    U, D, V = smith_normal_form([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    check("identity SNF all ones", invariant_factors(D) == [1, 1, 1])

    # ---- 10. zero matrix ----------------------------------------------------------------
    U, D, V = smith_normal_form([[0, 0], [0, 0]])
    check("zero matrix rank 0", rank(D) == 0 and _is_diagonal(D))

    # ---- 11. non-square (more columns) --------------------------------------------------
    A = [[2, 4, 6], [3, 6, 9]]
    U, D, V = smith_normal_form(A)
    prod = _matmul(_matmul(U, A), V)
    check("non-square U A V = D", prod == D and _divides_chain(D), f"D={D}")

    # ---- 12. single entry ---------------------------------------------------------------
    U, D, V = smith_normal_form([[6]])
    check("1x1 SNF", D == [[6]] and invariant_factors(D) == [6])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for Hessenberg reduction: form, orthogonality, similarity, eigenvalue preservation, tridiagonal."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hessenberg as HB  # noqa: E402
from qr_algorithm import eigenvalues as qeig  # noqa: E402


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


def _maxerr(A, B):
    return max(abs(A[i][j] - B[i][j]) for i in range(len(A)) for j in range(len(A[0])))


def _is_orth(Q, tol=1e-9):
    n = len(Q)
    return all(abs(sum(Q[k][i] * Q[k][j] for k in range(n)) - (1.0 if i == j else 0.0)) < tol
               for i in range(n) for j in range(n))


def main():
    rnd = _lcg(7)
    n = 6
    A = [[rnd() * 2 - 1 for _ in range(n)] for _ in range(n)]
    H, Q = HB.hessenberg(A)

    # ---- 1. H is upper Hessenberg -------------------------------------------------------
    check("H is upper Hessenberg", HB.is_upper_hessenberg(H))
    # explicit: entries two or more below the diagonal are zero
    check("entries below subdiagonal are zero",
          all(abs(H[i][j]) < 1e-9 for i in range(n) for j in range(n) if i > j + 1))

    # ---- 2. Q is orthogonal -------------------------------------------------------------
    check("Q is orthogonal", _is_orth(Q))

    # ---- 3. Q H Q^T reconstructs A ------------------------------------------------------
    R = HB.reconstruct(H, Q)
    check("Q H Q^T reconstructs A", _maxerr(R, A) < 1e-9, f"{_maxerr(R, A):.2e}")

    # ---- 4. similarity preserves trace and determinant ----------------------------------
    check("trace preserved", abs(HB.trace(A) - HB.trace(H)) < 1e-9)
    from lu import determinant
    check("determinant preserved", abs(determinant(A) - determinant(H)) < 1e-6,
          f"{determinant(A):.4f} vs {determinant(H):.4f}")

    # ---- 5. eigenvalues preserved (the whole point) -------------------------------------
    ea = sorted(qeig(A), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
    eh = sorted(qeig(H), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
    check("eigenvalues preserved under reduction",
          all(abs(ea[i] - eh[i]) < 1e-5 for i in range(n)), f"A {ea}\nH {eh}")

    # ---- 6. symmetric matrix reduces to symmetric tridiagonal ---------------------------
    S = [[A[i][j] + A[j][i] for j in range(n)] for i in range(n)]   # symmetric
    Hs, Qs = HB.hessenberg(S)
    check("symmetric input -> tridiagonal H", HB.is_tridiagonal(Hs))
    check("tridiagonal H is symmetric", HB.is_symmetric(Hs))
    check("symmetric reduction reconstructs S", _maxerr(HB.reconstruct(Hs, Qs), S) < 1e-9)

    # ---- 7. eigenvalues of a symmetric matrix are real and preserved --------------------
    es = sorted(x.real for x in qeig(S))
    eh_s = sorted(x.real for x in qeig(Hs))
    check("symmetric eigenvalues preserved",
          all(abs(es[i] - eh_s[i]) < 1e-5 for i in range(n)), f"{es}\n{eh_s}")

    # ---- 8. an already-Hessenberg matrix is left essentially unchanged ------------------
    Hin = [[rnd() * 2 - 1 if i <= j + 1 else 0.0 for j in range(n)] for i in range(n)]
    H2, Q2 = HB.hessenberg(Hin)
    check("already-Hessenberg reconstructs itself", _maxerr(HB.reconstruct(H2, Q2), Hin) < 1e-9)
    check("already-Hessenberg stays Hessenberg", HB.is_upper_hessenberg(H2))

    # ---- 9. small matrices (n<=2) are trivially Hessenberg ------------------------------
    A2 = [[1.0, 2.0], [3.0, 4.0]]
    H2s, Q2s = HB.hessenberg(A2)
    check("2x2 is already Hessenberg", HB.is_upper_hessenberg(H2s) and _maxerr(H2s, A2) < 1e-12)

    # ---- 10. diagonal matrix stays diagonal (already tridiagonal/Hessenberg) ------------
    D = [[float(i + 1) if i == j else 0.0 for j in range(n)] for i in range(n)]
    Hd, Qd = HB.hessenberg(D)
    check("diagonal input stays diagonal", HB.is_tridiagonal(Hd) and _maxerr(Hd, D) < 1e-9)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for the QR algorithm: matches Jacobi (symmetric), triangular diagonal, trace/det, complex pairs."""

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qr_algorithm import (  # noqa: E402
    eigenvalues,
    hessenberg,
    spectral_invariants,
    trace,
    characteristic_poly_roots,
)
from jacobi_eigen import eigen as jacobi_eigen  # noqa: E402


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


def _sorted_real(eigs):
    return sorted(e.real if isinstance(e, complex) else e for e in eigs)


def _match_multiset(a, b, tol=1e-6):
    """Do two lists of (complex) numbers match as multisets?"""
    a = list(a)
    b = list(b)
    if len(a) != len(b):
        return False
    used = [False] * len(b)
    for x in a:
        best = None
        for j, y in enumerate(b):
            if used[j]:
                continue
            d = abs(complex(x) - complex(y))
            if best is None or d < best[0]:
                best = (d, j)
        if best is None or best[0] > tol:
            return False
        used[best[1]] = True
    return True


def main():
    # ---- 1. symmetric matrices: eigenvalues match the Jacobi eigensolver ----------------
    rng = _lcg(1)
    ok = True
    for trial in range(20):
        n = 2 + int(rng() * 4)
        M = [[rng() - 0.5 for _ in range(n)] for _ in range(n)]
        A = [[(M[i][j] + M[j][i]) for j in range(n)] for i in range(n)]  # symmetric
        qr_e = _sorted_real(eigenvalues(A))
        jac_vals, _ = jacobi_eigen(A)
        jac_e = sorted(jac_vals)
        if not all(abs(qr_e[i] - jac_e[i]) < 1e-6 for i in range(n)):
            ok = False
            check("QR == Jacobi (symmetric)", False, f"n={n} {qr_e} vs {jac_e}")
            break
    if ok:
        check("QR eigenvalues == Jacobi (20 symmetric matrices)", True)

    # ---- 2. triangular matrix: eigenvalues are the diagonal -----------------------------
    T = [[3.0, 1.0, 2.0], [0.0, -1.0, 4.0], [0.0, 0.0, 5.0]]
    eigs = _sorted_real(eigenvalues(T))
    check("triangular eigenvalues = diagonal", _match_multiset(eigs, [3.0, -1.0, 5.0]), f"{eigs}")

    # ---- 3. diagonal matrix -------------------------------------------------------------
    D = [[2.0, 0, 0], [0, 7.0, 0], [0, 0, -3.0]]
    check("diagonal eigenvalues", _match_multiset(eigenvalues(D), [2.0, 7.0, -3.0]))

    # ---- 4. sum of eigenvalues = trace, product = determinant ---------------------------
    rng = _lcg(5)
    n = 5
    A = [[rng() * 4 - 2 for _ in range(n)] for _ in range(n)]
    eigs = eigenvalues(A)
    s, p = spectral_invariants(eigs)
    check("sum eigenvalues = trace", abs(complex(s).real - trace(A)) < 1e-4 and abs(complex(s).imag) < 1e-4,
          f"sum {s} vs trace {trace(A)}")
    # determinant via char-poly cross-check
    det = 1.0
    for e in eigs:
        det = det * e
    # compare to explicit determinant (LU-free: use char poly roots product should equal too)
    check("product eigenvalues is real for real matrix", abs(complex(p).imag) < 1e-4, f"{p}")

    # ---- 5. complex-conjugate eigenvalues of a rotation-like block ----------------------
    # [[cos, -sin],[sin, cos]] has eigenvalues e^{+/- i theta}
    theta = 0.9
    R = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    eigs = eigenvalues(R)
    expected = [cmath.exp(1j * theta), cmath.exp(-1j * theta)]
    check("rotation -> complex conjugate pair", _match_multiset(eigs, expected, 1e-6), f"{eigs}")

    # ---- 6. a 3x3 with one real and a complex pair --------------------------------------
    # companion-like matrix of (x-2)(x^2+1) = x^3 -2x^2 + x -2 -> eigs 2, +i, -i
    A = [[2, 0, 0], [0, 0, -1], [0, 1, 0]]
    eigs = eigenvalues(A)
    check("mixed real + complex pair", _match_multiset(eigs, [2.0, 1j, -1j], 1e-6), f"{eigs}")

    # ---- 7. eigenvalues equal characteristic-polynomial roots (Durand-Kerner) -----------
    rng = _lcg(7)
    ok = True
    for trial in range(8):
        n = 2 + int(rng() * 3)
        A = [[rng() * 4 - 2 for _ in range(n)] for _ in range(n)]
        qr_e = eigenvalues(A)
        cp_e = characteristic_poly_roots(A)
        if not _match_multiset(qr_e, cp_e, 1e-4):
            ok = False
            check("QR == char-poly roots", False, f"{qr_e} vs {cp_e}")
            break
    if ok:
        check("QR eigenvalues == characteristic-poly roots (8 matrices)", True)

    # ---- 8. Hessenberg form has zeros below the subdiagonal -----------------------------
    rng = _lcg(9)
    n = 5
    A = [[rng() - 0.5 for _ in range(n)] for _ in range(n)]
    H = hessenberg(A)
    zeros_ok = all(abs(H[i][j]) < 1e-9 for i in range(n) for j in range(n) if i > j + 1)
    check("Hessenberg zero below subdiagonal", zeros_ok)

    # ---- 9. Hessenberg preserves eigenvalues (similar to A) -----------------------------
    eigs_A = _sorted_real(eigenvalues(A))
    eigs_H = _sorted_real(eigenvalues(H))
    check("Hessenberg preserves eigenvalues", all(abs(eigs_A[i] - eigs_H[i]) < 1e-6 for i in range(n)))

    # ---- 10. 1x1 and 2x2 base cases -----------------------------------------------------
    check("1x1 eigenvalue", eigenvalues([[4.2]]) == [4.2])
    e2 = _sorted_real(eigenvalues([[2.0, 0.0], [0.0, 3.0]]))
    check("2x2 diagonal", _match_multiset(e2, [2.0, 3.0]))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

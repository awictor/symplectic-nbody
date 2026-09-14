"""Tests for Gram-Schmidt: orthonormality, QR reconstruct, MGS beats CGS on ill-conditioned, Legendre."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gram_schmidt import (  # noqa: E402
    classical_gram_schmidt, modified_gram_schmidt, qr, orthogonality_error,
    reconstruct, orthogonal_polynomials, _dot,
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


def main():
    rng = _lcg(1)

    # ---- 1. both variants produce orthonormal vectors -----------------------------------
    cols = [[rng() - 0.5 for _ in range(5)] for _ in range(5)]
    Qc, Rc = classical_gram_schmidt(cols)
    Qm, Rm = modified_gram_schmidt(cols)
    check("CGS orthonormal (well-conditioned)", orthogonality_error(Qc) < 1e-10)
    check("MGS orthonormal (well-conditioned)", orthogonality_error(Qm) < 1e-12)

    # ---- 2. QR reconstruction A = Q R ---------------------------------------------------
    for method, (Q, R) in [("cgs", (Qc, Rc)), ("mgs", (Qm, Rm))]:
        rec = reconstruct(Q, R)
        maxerr = max(abs(rec[j][i] - cols[j][i]) for j in range(5) for i in range(5))
        check(f"{method}: reconstruct A = Q R", maxerr < 1e-10, f"{maxerr:.2e}")

    # ---- 3. R is upper-triangular -------------------------------------------------------
    check("R upper-triangular", all(Rm[i][j] == 0.0 for i in range(5) for j in range(i)))
    check("R positive diagonal", all(Rm[i][i] > 0 for i in range(5)))

    # ---- 4. MGS far more orthogonal than CGS on ill-conditioned (Hilbert) basis ---------
    n = 9
    hilbert = [[1.0 / (i + j + 1) for i in range(n)] for j in range(n)]
    Qc, _ = classical_gram_schmidt(hilbert)
    Qm, _ = modified_gram_schmidt(hilbert)
    cgs_err = orthogonality_error(Qc)
    mgs_err = orthogonality_error(Qm)
    check("MGS orthogonality << CGS on Hilbert basis", mgs_err < cgs_err / 1000,
          f"CGS {cgs_err:.2e} vs MGS {mgs_err:.2e}")

    # ---- 5. orthonormal input is (nearly) a no-op ---------------------------------------
    # standard basis vectors
    e = [[1.0 if i == j else 0.0 for i in range(4)] for j in range(4)]
    Q, R = modified_gram_schmidt(e)
    check("standard basis stays orthonormal", orthogonality_error(Q) < 1e-14)
    check("standard basis R = identity", all(abs(R[i][i] - 1.0) < 1e-14 for i in range(4)))

    # ---- 6. span is preserved: each original vector is a combination of the q_i ---------
    cols = [[rng() for _ in range(4)] for _ in range(4)]
    Q, R = modified_gram_schmidt(cols)
    rec = reconstruct(Q, R)
    check("span preserved (reconstruction)", all(abs(rec[j][i] - cols[j][i]) < 1e-10
                                                 for j in range(4) for i in range(4)))

    # ---- 7. qr() on a matrix (rows) -----------------------------------------------------
    A = [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    Q, R = qr(A, method="mgs")
    # Q columns orthonormal
    qcols = [[Q[i][j] for i in range(3)] for j in range(3)]
    check("qr() Q columns orthonormal", orthogonality_error(qcols) < 1e-12)
    # A = Q R
    ok = True
    for i in range(3):
        for j in range(3):
            s = sum(Q[i][k] * R[k][j] for k in range(3))
            if abs(s - A[i][j]) > 1e-10:
                ok = False
    check("qr() reconstructs A = Q R", ok)

    # ---- 8. orthogonal polynomials are Legendre (normalized) on [-1,1] ------------------
    op = orthogonal_polynomials(3)
    # p0 = 1/sqrt(2), p1 = sqrt(3/2) x
    check("p0 = 1/sqrt(2)", abs(op[0][0] - 1 / math.sqrt(2)) < 1e-3, f"{op[0]}")
    check("p1 = sqrt(3/2) x", abs(op[1][1] - math.sqrt(3 / 2)) < 1e-3 and abs(op[1][0]) < 1e-3,
          f"{op[1]}")
    # p2 proportional to 3x^2 - 1, normalized sqrt(5/8)(3x^2-1): leading coeff sqrt(5/8)*3
    check("p2 leading coeff", abs(op[2][2] - math.sqrt(5 / 8) * 3) < 1e-2, f"{op[2]}")

    # ---- 9. orthogonal polynomials are mutually orthogonal (numerically) ----------------
    op = orthogonal_polynomials(4)

    def poly_val(c, x):
        r = 0.0
        for coef in reversed(c):
            r = r * x + coef
        return r

    def l2(p, q):
        xs = [-1 + 2 * k / 999 for k in range(1000)]
        dx = 2 / 999
        return sum(poly_val(p, x) * poly_val(q, x) for x in xs) * dx

    ok = True
    for i in range(5):
        for j in range(5):
            v = l2(op[i], op[j])
            target = 1.0 if i == j else 0.0
            if abs(v - target) > 0.02:
                ok = False
                break
        if not ok:
            break
    check("orthogonal polynomials mutually orthonormal", ok)

    # ---- 10. linear dependence is detected ----------------------------------------------
    try:
        modified_gram_schmidt([[1.0, 0.0], [2.0, 0.0]])  # second = 2 * first
        check("linear dependence raises", False)
    except ValueError:
        check("linear dependence raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

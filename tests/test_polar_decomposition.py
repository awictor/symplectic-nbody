"""Tests for polar decomposition: A=UP, orthogonality, SPD factor, SVD vs Newton, closest rotation."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import polar_decomposition as PD  # noqa: E402
import svd as _svd  # noqa: E402
from jacobi_eigen import sorted_eigen  # noqa: E402


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


def main():
    rnd = _lcg(7)
    n = 4
    A = [[rnd() * 2 - 1 for _ in range(n)] for _ in range(n)]

    # ---- 1. Newton: U orthogonal, P symmetric, U P == A ---------------------------------
    U, P = PD.polar_newton(A)
    check("Newton U is orthogonal", PD.is_orthogonal(U, tol=1e-9))
    check("Newton P is symmetric", PD.is_symmetric(P, tol=1e-9))
    UP = PD._matmul(U, P)
    check("Newton U P reconstructs A", _maxerr(UP, A) < 1e-9, f"{_maxerr(UP, A):.2e}")

    # ---- 2. P is positive semidefinite (nonneg eigenvalues) -----------------------------
    vals, _ = sorted_eigen(P)
    check("P is positive semidefinite", all(v > -1e-9 for v in vals), f"min eig {min(vals):.2e}")

    # ---- 3. P eigenvalues equal the singular values of A --------------------------------
    _, S, _ = _svd.svd(A)
    sv = sorted(S, reverse=True)
    pe = sorted(vals, reverse=True)
    check("P eigenvalues == singular values of A",
          all(abs(pe[i] - sv[i]) < 1e-4 for i in range(len(sv))), f"P {pe} S {sv}")

    # ---- 4. SVD route agrees with Newton (to SVD accuracy) ------------------------------
    Us, Ps = PD.polar_svd(A)
    check("SVD and Newton polar factors agree", _maxerr(Us, U) < 1e-4, f"{_maxerr(Us, U):.2e}")
    check("SVD U P reconstructs A", _maxerr(PD._matmul(Us, Ps), A) < 1e-4)

    # ---- 5. an already-orthogonal matrix has P = I --------------------------------------
    th = 0.7
    Q = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
    Uq, Pq = PD.polar_newton(Q)
    check("orthogonal input: P == I",
          _maxerr(Pq, [[1.0, 0.0], [0.0, 1.0]]) < 1e-9, f"{_maxerr(Pq, [[1,0],[0,1]]):.2e}")
    check("orthogonal input: U == Q", _maxerr(Uq, Q) < 1e-9)

    # ---- 6. U is the CLOSEST orthogonal matrix (Frobenius) ------------------------------
    Ubest = PD.closest_orthogonal(A)

    def frob_dist(M):
        return math.sqrt(sum((M[i][j] - A[i][j]) ** 2 for i in range(n) for j in range(n)))

    d_polar = frob_dist(Ubest)
    # test several random orthogonal matrices (built by polar-izing random matrices)
    ok = True
    rr = _lcg(99)
    for _ in range(20):
        R = [[rr() * 2 - 1 for _ in range(n)] for _ in range(n)]
        Q2 = PD.closest_orthogonal(R)             # some orthogonal matrix
        if frob_dist(Q2) < d_polar - 1e-7:
            ok = False
    check("polar U is the closest orthogonal matrix", ok, f"d_polar {d_polar:.4f}")

    # ---- 7. determinant of U is +/-1 (orthogonal) ---------------------------------------
    from lu import determinant
    detU = determinant(U)
    check("det(U) == +/-1", abs(abs(detU) - 1.0) < 1e-8, f"{detU:.6f}")

    # ---- 8. a reflection (det A < 0) is handled: det(U) < 0 -----------------------------
    Aref = [[1.0, 0.0], [0.0, -2.0]]              # det = -2 < 0
    Ur, Pr = PD.polar_newton(Aref)
    check("reflection reconstructs A", _maxerr(PD._matmul(Ur, Pr), Aref) < 1e-9)
    check("reflection: det(U) < 0", determinant(Ur) < 0, f"{determinant(Ur)}")

    # ---- 9. pure stretch (symmetric PD A): U == I, P == A -------------------------------
    Astretch = [[3.0, 1.0], [1.0, 2.0]]           # symmetric PD
    Ustr, Pstr = PD.polar_newton(Astretch)
    check("SPD input: U == I", _maxerr(Ustr, [[1.0, 0.0], [0.0, 1.0]]) < 1e-8)
    check("SPD input: P == A", _maxerr(Pstr, Astretch) < 1e-8)

    # ---- 10. re-orthogonalize a drifted rotation ----------------------------------------
    drift = [[math.cos(th) + 0.02, -math.sin(th) - 0.01],
             [math.sin(th) + 0.015, math.cos(th) + 0.005]]
    Ud = PD.closest_orthogonal(drift)
    check("re-orthogonalized drifted rotation is orthogonal", PD.is_orthogonal(Ud, tol=1e-9))
    # and it should be close to the clean rotation
    check("re-orthogonalized close to the true rotation", _maxerr(Ud, Q) < 0.05,
          f"{_maxerr(Ud, Q):.4f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for Sherman-Morrison-Woodbury: inverse/solve/det updates vs brute-force re-inversion."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sherman_morrison as SM  # noqa: E402
from lu import inverse, determinant  # noqa: E402
from linsolve import solve  # noqa: E402


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
    n = 6
    # diagonally dominant -> well conditioned
    A = [[rnd() * 2 - 1 + (4 if i == j else 0) for j in range(n)] for i in range(n)]
    A_inv = inverse(A)
    u = [rnd() for _ in range(n)]
    v = [rnd() for _ in range(n)]
    mod = [[A[i][j] + u[i] * v[j] for j in range(n)] for i in range(n)]

    # ---- 1. rank-1 inverse update matches full re-inversion -----------------------------
    sm = SM.sherman_morrison_inverse(A_inv, u, v)
    full = inverse(mod)
    check("Sherman-Morrison inverse == full re-inversion", _maxerr(sm, full) < 1e-9,
          f"{_maxerr(sm, full):.2e}")

    # ---- 2. the updated inverse actually inverts the modified matrix --------------------
    prod = [[sum(mod[i][k] * sm[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    ident_err = max(abs(prod[i][j] - (1.0 if i == j else 0.0)) for i in range(n) for j in range(n))
    check("(A+uv^T) * SM-inverse == I", ident_err < 1e-9, f"{ident_err:.2e}")

    # ---- 3. determinant lemma matches a direct determinant ------------------------------
    dl = SM.determinant_lemma(A, u, v)
    check("determinant lemma matches det(A+uv^T)", abs(dl - determinant(mod)) < 1e-6,
          f"{dl} vs {determinant(mod)}")

    # ---- 4. rank-1 solve update matches solving the modified system directly ------------
    b = [rnd() for _ in range(n)]
    xs = SM.sherman_morrison_solve(A_inv, u, v, b)
    xfull = solve(mod, b)
    check("Sherman-Morrison solve == direct solve",
          max(abs(xs[i] - xfull[i]) for i in range(n)) < 1e-9)

    # ---- 5. Woodbury rank-k inverse update matches full re-inversion --------------------
    k = 3
    U = [[rnd() for _ in range(k)] for _ in range(n)]
    V = [[rnd() for _ in range(k)] for _ in range(n)]
    C = [[rnd() * 2 - 1 + (3 if i == j else 0) for j in range(k)] for i in range(k)]
    # build U C V^T
    CVt = [[sum(C[a][bb] * V[j][bb] for bb in range(k)) for j in range(n)] for a in range(k)]
    UCVt = [[sum(U[i][a] * CVt[a][j] for a in range(k)) for j in range(n)] for i in range(n)]
    mod2 = [[A[i][j] + UCVt[i][j] for j in range(n)] for i in range(n)]
    wb = SM.woodbury_inverse(A_inv, U, C, V)
    full2 = inverse(mod2)
    check("Woodbury rank-3 inverse == full re-inversion", _maxerr(wb, full2) < 1e-9,
          f"{_maxerr(wb, full2):.2e}")

    # ---- 6. Woodbury with k=1 agrees with Sherman-Morrison ------------------------------
    U1 = [[u[i]] for i in range(n)]
    V1 = [[v[i]] for i in range(n)]
    C1 = [[1.0]]
    wb1 = SM.woodbury_inverse(A_inv, U1, C1, V1)
    check("Woodbury (k=1) == Sherman-Morrison", _maxerr(wb1, sm) < 1e-9)

    # ---- 7. chaining several rank-1 updates == one big re-inversion ---------------------
    cur_inv = [row[:] for row in A_inv]
    cur_mat = [row[:] for row in A]
    for t in range(4):
        uu = [rnd() for _ in range(n)]
        vv = [rnd() * 0.3 for _ in range(n)]      # small so it stays invertible
        cur_inv = SM.sherman_morrison_inverse(cur_inv, uu, vv)
        cur_mat = [[cur_mat[i][j] + uu[i] * vv[j] for j in range(n)] for i in range(n)]
    check("chained rank-1 updates == full inverse of the final matrix",
          _maxerr(cur_inv, inverse(cur_mat)) < 1e-7, f"{_maxerr(cur_inv, inverse(cur_mat)):.2e}")

    # ---- 8. a singular update is flagged ------------------------------------------------
    # choose u, v so that 1 + v^T A^{-1} u = 0: pick u, then v scaled to hit -1
    Ainv_u = [sum(A_inv[i][j] * u[j] for j in range(n)) for i in range(n)]
    # start from an arbitrary w, set v = w then rescale so v^T (A^{-1} u) = -1
    w = [1.0] + [0.0] * (n - 1)
    dot = sum(w[i] * Ainv_u[i] for i in range(n))
    if abs(dot) > 1e-9:
        v_sing = [wi * (-1.0 / dot) for wi in w]
        try:
            SM.sherman_morrison_inverse(A_inv, u, v_sing)
            check("singular rank-1 update raises", False, "no exception")
        except ValueError:
            check("singular rank-1 update raises", True)
    else:
        check("singular rank-1 update raises", True)   # degenerate setup, skip

    # ---- 9. update by a zero outer product leaves the inverse unchanged ------------------
    z = [0.0] * n
    sm_z = SM.sherman_morrison_inverse(A_inv, z, v)
    check("zero update leaves inverse unchanged", _maxerr(sm_z, A_inv) < 1e-12)

    # ---- 10. solve update matches inverse-then-multiply ---------------------------------
    x_via_inv = [sum(sm[i][j] * b[j] for j in range(n)) for i in range(n)]
    check("solve update == updated-inverse times b",
          max(abs(xs[i] - x_via_inv[i]) for i in range(n)) < 1e-9)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

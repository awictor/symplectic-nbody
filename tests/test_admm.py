"""Tests for ADMM: LASSO vs FISTA, NNLS KKT, residual decrease, constrained LS, rho adaptation."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import admm as AD  # noqa: E402
import fista as F  # noqa: E402


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


def main():
    rnd = _lcg(5)
    n, p = 60, 12
    A = [[rnd() * 2 - 1 for _ in range(p)] for _ in range(n)]
    xtrue = [2.0, 0, 0, -1.5, 0, 0, 3.0, 0, 0, -0.5, 0, 0]
    b = [sum(A[i][j] * xtrue[j] for j in range(p)) for i in range(n)]

    # ---- 1. ADMM LASSO matches the repo's FISTA solver ----------------------------------
    lam = 0.5
    xa, ha = AD.lasso(A, b, lam, rho=1.0, max_iter=8000, tol=1e-11)
    xf = F.lasso_fista(A, b, lam, max_iter=20000)
    check("ADMM LASSO == FISTA LASSO", max(abs(xa[j] - xf[j]) for j in range(p)) < 1e-3,
          f"admm {[round(v,3) for v in xa]} fista {[round(v,3) for v in xf]}")

    # ---- 2. residuals decrease to (near) zero -------------------------------------------
    r_final, s_final = ha[-1]
    check("primal residual -> 0", r_final < 1e-6, f"{r_final}")
    check("dual residual -> 0", s_final < 1e-6, f"{s_final}")
    check("primal residual decreased from start", ha[-1][0] < ha[0][0])

    # ---- 3. LASSO recovers the sparse structure -----------------------------------------
    zeros_ok = all(abs(xa[j]) < 0.05 for j in range(p) if xtrue[j] == 0)
    check("ADMM LASSO zeros inactive features", zeros_ok, f"{[round(v,3) for v in xa]}")

    # ---- 4. NNLS: non-negative and KKT --------------------------------------------------
    rnd = _lcg(77)
    m, q = 40, 6
    An = [[rnd() * 2 - 1 for _ in range(q)] for _ in range(m)]
    bn = [rnd() * 3 for _ in range(m)]
    xn, hn = AD.nnls(An, bn, max_iter=8000, tol=1e-11)
    check("NNLS respects x >= 0", all(v >= -1e-7 for v in xn), f"{[round(v,3) for v in xn]}")
    r = [sum(An[i][j] * xn[j] for j in range(q)) - bn[i] for i in range(m)]
    g = AD._matTvec(An, r)
    kkt = all((abs(g[j]) < 1e-2) if xn[j] > 1e-5 else (g[j] > -1e-2) for j in range(q))
    check("NNLS satisfies KKT conditions", kkt, f"g {[round(v,3) for v in g]}")

    # ---- 5. cross-check NNLS against FISTA's NNLS ---------------------------------------
    xn_f = F.nnls_fista(An, bn, max_iter=20000)
    check("ADMM NNLS == FISTA NNLS", max(abs(xn[j] - xn_f[j]) for j in range(q)) < 1e-2,
          f"admm {[round(v,3) for v in xn]} fista {[round(v,3) for v in xn_f]}")

    # ---- 6. generic ADMM with explicit prox operators solves consensus ------------------
    # min (1/2)||x - a||^2 + (1/2)||x - c||^2  (via f, g); optimum is (a + c)/2
    a = [1.0, 4.0, -2.0]
    c = [3.0, 0.0, 6.0]

    def prox_f(v, rho):
        # argmin (1/2)||x-a||^2 + (rho/2)||x-v||^2 = (a + rho v)/(1+rho)
        return [(a[i] + rho * v[i]) / (1 + rho) for i in range(3)]

    def prox_g(v, rho):
        return [(c[i] + rho * v[i]) / (1 + rho) for i in range(3)]

    x, hist = AD.admm(prox_f, prox_g, 3, rho=1.0, max_iter=2000, tol=1e-12)
    expected = [(a[i] + c[i]) / 2 for i in range(3)]
    check("generic ADMM consensus == (a+c)/2",
          max(abs(x[i] - expected[i]) for i in range(3)) < 1e-6, f"{x} vs {expected}")

    # ---- 7. rho adaptation still converges to the same optimum --------------------------
    xad, had = AD.adaptive_lasso(A, b, lam, max_iter=8000, tol=1e-10)
    check("adaptive-rho LASSO == FISTA", max(abs(xad[j] - xf[j]) for j in range(p)) < 2e-3,
          f"{[round(v,3) for v in xad]}")
    check("adaptive LASSO converges", had[-1][0] < 1e-5 and had[-1][1] < 1e-5)

    # ---- 8. lambda -> 0 recovers the exact linear fit -----------------------------------
    x0, _ = AD.lasso(A, b, 1e-7, rho=1.0, max_iter=20000, tol=1e-12)
    check("lam->0 recovers exact fit", max(abs(x0[j] - xtrue[j]) for j in range(p)) < 0.05,
          f"{[round(v,3) for v in x0]}")

    # ---- 9. larger lambda gives a sparser solution --------------------------------------
    x_small, _ = AD.lasso(A, b, 0.1, max_iter=8000)
    x_large, _ = AD.lasso(A, b, 3.0, max_iter=8000)
    nnz_small = sum(1 for v in x_small if abs(v) > 1e-4)
    nnz_large = sum(1 for v in x_large if abs(v) > 1e-4)
    check("larger lambda -> sparser", nnz_large <= nnz_small, f"small {nnz_small} large {nnz_large}")

    # ---- 10. constrained LS: NNLS with an all-positive true solution recovers it --------
    Ap = [[rnd() * 2 - 1 for _ in range(4)] for _ in range(50)]
    xpos = [1.0, 2.5, 0.3, 4.0]
    bp = [sum(Ap[i][j] * xpos[j] for j in range(4)) for i in range(50)]
    xr, _ = AD.nnls(Ap, bp, max_iter=8000, tol=1e-11)
    check("NNLS recovers an all-positive solution",
          max(abs(xr[j] - xpos[j]) for j in range(4)) < 1e-3, f"{[round(v,3) for v in xr]}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

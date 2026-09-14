"""Tests for FISTA: quadratic exactness, soft-threshold, Lasso recovery, ISTA vs FISTA acceleration, NNLS KKT."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
    # ---- 1. scalar soft-threshold is the prox of t|.| -----------------------------------
    check("soft_threshold shrinks and clamps",
          F.soft_threshold(3, 1) == 2 and F.soft_threshold(0.5, 1) == 0.0
          and F.soft_threshold(-4, 1.5) == -2.5)
    # prox_l1 vectorized
    check("prox_l1 element-wise", F.prox_l1([3, -0.5, 2], 1) == [2, 0.0, 1])

    # ---- 2. FISTA on a smooth quadratic reaches the exact minimizer ---------------------
    c = [3.0, -2.0, 5.0, 1.0]
    f = lambda x: 0.5 * sum((x[i] - c[i]) ** 2 for i in range(len(c)))
    grad = lambda x: [x[i] - c[i] for i in range(len(c))]
    x, _ = F.fista(f, grad, F.prox_zero, [0.0] * len(c), step=1.0, max_iter=500)
    check("FISTA minimizes a quadratic exactly", max(abs(x[i] - c[i]) for i in range(len(c))) < 1e-6,
          f"{x}")

    # ---- 3. Lasso recovers a planted sparse signal --------------------------------------
    rnd = _lcg(5)
    n, p = 60, 12
    A = [[rnd() * 2 - 1 for _ in range(p)] for _ in range(n)]
    xtrue = [2.0, 0, 0, -1.5, 0, 0, 3.0, 0, 0, -0.5, 0, 0]
    b = [sum(A[i][j] * xtrue[j] for j in range(p)) for i in range(n)]
    xhat = F.lasso_fista(A, b, lam=0.2, max_iter=8000)
    # the zero coordinates of xtrue should be (near) zero in xhat
    zeros_ok = all(abs(xhat[j]) < 0.05 for j in range(p) if xtrue[j] == 0)
    nonzero_ok = all(abs(xhat[j] - xtrue[j]) < 0.15 for j in range(p) if xtrue[j] != 0)
    check("Lasso zeros the inactive features", zeros_ok, f"{[round(v,3) for v in xhat]}")
    check("Lasso approximately recovers active features", nonzero_ok, f"{[round(v,3) for v in xhat]}")

    # ---- 4. FISTA Lasso == ISTA Lasso (same objective, same optimum) --------------------
    lam = 0.3

    def f_lasso(x):
        r = [sum(A[i][j] * x[j] for j in range(p)) - b[i] for i in range(n)]
        return 0.5 * sum(v * v for v in r) + lam * sum(abs(v) for v in x)

    def g_lasso(x):
        r = [sum(A[i][j] * x[j] for j in range(p)) - b[i] for i in range(n)]
        return F._matTvec(A, r)

    L = F._spectral_norm_sq(A)
    step = 1.0 / L
    prox = lambda z, t: F.prox_l1(z, t * lam)
    xf, hf = F.fista(f_lasso, g_lasso, prox, [0.0] * p, step, max_iter=10000, tol=1e-13)
    xi, hi = F.proximal_gradient(f_lasso, g_lasso, prox, [0.0] * p, step, max_iter=10000, tol=1e-13)
    check("FISTA and ISTA reach the same Lasso optimum",
          abs(hf[-1] - hi[-1]) < 1e-6 and max(abs(xf[j] - xi[j]) for j in range(p)) < 1e-3,
          f"fista {hf[-1]:.6f} ista {hi[-1]:.6f}")

    # ---- 5. FISTA reaches a target accuracy in fewer iterations than ISTA ---------------
    fstar = min(min(hf), min(hi))

    def iters_to(hist, eps):
        for k, v in enumerate(hist):
            if v - fstar < eps:
                return k
        return len(hist)

    kf = iters_to(hf, 1e-4)
    ki = iters_to(hi, 1e-4)
    check("FISTA accelerates over ISTA (fewer iters to 1e-4)", kf < ki, f"fista {kf} ista {ki}")
    # and the gap should be substantial on an ill-conditioned problem
    kf2 = iters_to(hf, 1e-6)
    ki2 = iters_to(hi, 1e-6)
    check("FISTA advantage grows at tighter tolerance", kf2 < ki2, f"fista {kf2} ista {ki2}")

    # ---- 6. objective is (eventually) driven down; FISTA history ends near optimum ------
    check("FISTA drives the objective to the optimum", abs(hf[-1] - fstar) < 1e-6)

    # ---- 7. non-negative least squares: constraint + KKT --------------------------------
    rnd = _lcg(77)
    m, q = 40, 6
    An = [[rnd() * 2 - 1 for _ in range(q)] for _ in range(m)]
    bn = [rnd() * 3 for _ in range(m)]
    xn = F.nnls_fista(An, bn, max_iter=10000)
    check("NNLS respects x >= 0", all(v >= -1e-7 for v in xn), f"{[round(v,3) for v in xn]}")
    # KKT: gradient >= 0 where x==0, gradient ~0 where x>0
    r = [sum(An[i][j] * xn[j] for j in range(q)) - bn[i] for i in range(m)]
    g = F._matTvec(An, r)
    kkt = all((abs(g[j]) < 1e-2) if xn[j] > 1e-5 else (g[j] > -1e-2) for j in range(q))
    check("NNLS satisfies KKT conditions", kkt, f"g {[round(v,3) for v in g]}")

    # ---- 8. backtracking FISTA reaches the same optimum as fixed-step FISTA -------------
    xb, hb = F.backtracking_fista(f_lasso, g_lasso, prox, [0.0] * p, L0=0.5, max_iter=3000, tol=1e-12)
    check("backtracking FISTA matches fixed-step optimum",
          abs(f_lasso(xb) - hf[-1]) < 1e-4, f"bt {f_lasso(xb):.6f} fista {hf[-1]:.6f}")

    # ---- 9. lambda -> larger gives a sparser solution -----------------------------------
    x_small = F.lasso_fista(A, b, lam=0.1, max_iter=6000)
    x_large = F.lasso_fista(A, b, lam=3.0, max_iter=6000)
    nnz_small = sum(1 for v in x_small if abs(v) > 1e-4)
    nnz_large = sum(1 for v in x_large if abs(v) > 1e-4)
    check("larger lambda gives a sparser solution", nnz_large <= nnz_small,
          f"nnz small {nnz_small} large {nnz_large}")

    # ---- 10. zero regularization Lasso == ordinary least squares (well-determined) ------
    # with lam ~ 0 and a consistent system, FISTA should recover xtrue closely
    x0 = F.lasso_fista(A, b, lam=1e-6, max_iter=20000)
    check("lam->0 recovers the exact linear fit",
          max(abs(x0[j] - xtrue[j]) for j in range(p)) < 0.05, f"{[round(v,3) for v in x0]}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

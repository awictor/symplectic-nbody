"""Tests for levinson_durbin: Toeplitz solve vs dense Gauss, AR recovery, reflection stability."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from levinson_durbin import (solve_toeplitz, autocorrelation, levinson_durbin,  # noqa: E402
                             ar_fit, ar_predict, is_positive_definite)


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


def dense_solve(A, b):
    """Reference: solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pivot = M[col][col]
        for r in range(n):
            if r != col:
                factor = M[r][col] / pivot
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def toeplitz_matrix(r, n):
    return [[r[abs(i - j)] for j in range(n)] for i in range(n)]


def matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0


def main():
    # ---- 1. hand-checked 3x3 Toeplitz system ------------------------------------------
    # T = [[4,1,0],[1,4,1],[0,1,4]] (r = [4,1,0]), b = [1,2,3]
    r = [4.0, 1.0, 0.0]
    b = [1.0, 2.0, 3.0]
    x = solve_toeplitz(r, b)
    ref = dense_solve(toeplitz_matrix(r, 3), b)
    check("3x3 Toeplitz solve matches dense", all(abs(a - c) < 1e-9 for a, c in zip(x, ref)),
          f"{x} vs {ref}")

    # ---- 2. random positive-definite Toeplitz systems vs dense ------------------------
    rng = LCG(7)
    for trial in range(6):
        n = 5 + trial
        # build a PD Toeplitz: r[k] = rho^k * scale gives a valid autocorrelation (AR(1))
        rho = 0.3 + 0.4 * rng.u()
        scale = 1.0 + rng.u()
        rr = [scale * rho ** k for k in range(n)]
        bb = [rng.normal() for _ in range(n)]
        xs = solve_toeplitz(rr, bb)
        A = toeplitz_matrix(rr, n)
        xref = dense_solve(A, bb)
        agree = all(abs(a - c) < 1e-6 for a, c in zip(xs, xref))
        check(f"random PD Toeplitz solve matches dense (n={n})", agree,
              f"max diff {max(abs(a-c) for a,c in zip(xs,xref)):.2e}")
        # residual small
        resid = matvec(A, xs)
        rmax = max(abs(resid[i] - bb[i]) for i in range(n))
        check(f"residual T x - b ~ 0 (n={n})", rmax < 1e-6, f"resid {rmax:.2e}")

    # ---- 3. AR recovery on synthesised data -------------------------------------------
    # generate x_t = 0.75 x_{t-1} - 0.5 x_{t-2} + e_t
    true_a = [0.75, -0.5]
    gen = LCG(2024)
    N = 6000
    xs = [0.0, 0.0]
    for t in range(2, N):
        e = 0.3 * gen.normal()
        xs.append(true_a[0] * xs[t - 1] + true_a[1] * xs[t - 2] + e)
    coeffs, reflection, err = ar_fit(xs, 2)
    check("AR(2) coefficient 1 recovered", abs(coeffs[0] - true_a[0]) < 0.05,
          f"{coeffs[0]:.4f} vs {true_a[0]}")
    check("AR(2) coefficient 2 recovered", abs(coeffs[1] - true_a[1]) < 0.05,
          f"{coeffs[1]:.4f} vs {true_a[1]}")
    check("reflection coefficients all |k|<1 (stable)", all(abs(k) < 1 for k in reflection),
          f"{reflection}")

    # error variance ~ variance of actual one-step residuals
    resid = []
    for t in range(2, N):
        pred = coeffs[0] * xs[t - 1] + coeffs[1] * xs[t - 2]
        resid.append(xs[t] - pred)
    var_resid = sum(e * e for e in resid) / len(resid)
    check("reported error variance matches residual variance",
          abs(err - var_resid) / var_resid < 0.05, f"err={err:.4f} residvar={var_resid:.4f}")

    # ---- 4. one-step predictor -------------------------------------------------------
    nxt = ar_predict(xs, coeffs)
    hand = coeffs[0] * xs[-1] + coeffs[1] * xs[-2]
    check("ar_predict matches manual dot product", abs(nxt - hand) < 1e-12)

    # ---- 5. positive-definiteness test -----------------------------------------------
    pd = [2.0, 0.5, 0.1]      # AR-like, should be PD
    check("PD Toeplitz flagged positive definite", is_positive_definite(pd))
    notpd = [1.0, 1.2, 0.0]   # off-diagonal too large -> reflection |k|>=1
    check("non-PD Toeplitz flagged not positive definite", not is_positive_definite(notpd))

    # ---- 6. autocorrelation basics ---------------------------------------------------
    ac = autocorrelation([1.0, 2.0, 3.0, 4.0, 5.0], 2)
    check("autocorrelation r[0] is the variance", abs(ac[0] - 2.0) < 1e-9, f"{ac[0]}")
    check("autocorrelation decreasing in lag", ac[0] > ac[1] > ac[2])

    # ---- 7. AR(1) exact reflection ---------------------------------------------------
    # for r = [1, rho, rho^2] an AR(1), first reflection coeff = -rho, second ~ 0
    rho = 0.6
    rr = [1.0, rho, rho ** 2, rho ** 3]
    _, refl, _ = levinson_durbin(rr, 3)
    check("AR(1) first reflection = -rho", abs(refl[0] + rho) < 1e-9, f"{refl[0]}")
    check("AR(1) higher reflections ~ 0", abs(refl[1]) < 1e-9 and abs(refl[2]) < 1e-9,
          f"{refl[1:]}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

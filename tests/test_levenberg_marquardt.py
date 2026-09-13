"""Tests for Levenberg-Marquardt: recovers exp/Gaussian/sine params, linear==normal eqns, monotone."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from levenberg_marquardt import (  # noqa: E402
    levenberg_marquardt,
    make_residual,
    exp_model,
    gaussian_model,
    sine_model,
    sum_of_squares,
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


def main():
    # ---- 1. recover exponential model parameters (noiseless) ----------------------------
    true_p = [2.5, -0.7, 1.0]
    xs = [i * 0.2 for i in range(30)]
    ys = [exp_model(x, true_p) for x in xs]
    res = make_residual(exp_model, xs, ys)
    out = levenberg_marquardt(res, [1.0, -0.3, 0.5])
    check("exp: recovers parameters", all(abs(out["params"][i] - true_p[i]) < 1e-4 for i in range(3)),
          f"{[round(v,4) for v in out['params']]}")
    check("exp: cost near zero", out["cost"] < 1e-8, f"{out['cost']:.2e}")
    check("exp: converged flag", out["converged"])

    # ---- 2. recover Gaussian peak -------------------------------------------------------
    true_g = [3.0, 1.5, 0.8]
    xs = [i * 0.1 - 1 for i in range(50)]
    ys = [gaussian_model(x, true_g) for x in xs]
    res = make_residual(gaussian_model, xs, ys)
    out = levenberg_marquardt(res, [2.0, 1.0, 1.0])
    check("gaussian: recovers parameters",
          all(abs(out["params"][i] - true_g[i]) < 1e-3 for i in range(3)),
          f"{[round(v,4) for v in out['params']]}")

    # ---- 3. recover sinusoid ------------------------------------------------------------
    true_s = [1.5, 2.0, 0.5]
    xs = [i * 0.05 for i in range(80)]
    ys = [sine_model(x, true_s) for x in xs]
    res = make_residual(sine_model, xs, ys)
    out = levenberg_marquardt(res, [1.0, 2.1, 0.3])
    # sine params can land in an equivalent phase; check the fit reproduces the data
    fit_err = max(abs(sine_model(xs[i], out["params"]) - ys[i]) for i in range(len(xs)))
    check("sine: fit reproduces data", fit_err < 1e-4, f"max err {fit_err:.2e}")

    # ---- 4. linear model: LM must match the analytic normal-equations solution ----------
    # model a*x + b; fit to data with known least-squares answer
    xs = [0, 1, 2, 3, 4, 5]
    ys = [1.0, 3.1, 4.9, 7.2, 8.8, 11.1]  # ~ 2x + 1

    def lin(x, p):
        return p[0] * x + p[1]

    res = make_residual(lin, xs, ys)
    out = levenberg_marquardt(res, [0.0, 0.0])
    # analytic least squares
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(xs[i] * ys[i] for i in range(n))
    a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - a * sx) / n
    check("linear: matches normal equations",
          abs(out["params"][0] - a) < 1e-6 and abs(out["params"][1] - b) < 1e-6,
          f"LM {out['params']} vs OLS [{a:.4f}, {b:.4f}]")

    # ---- 5. cost decreases monotonically over accepted steps ----------------------------
    hist = out["history"]
    check("cost history non-increasing", all(hist[i + 1] <= hist[i] + 1e-12 for i in range(len(hist) - 1)))

    # ---- 6. analytic Jacobian gives the same fit as finite differences ------------------
    true_p = [2.0, -0.5, 0.3]
    xs = [i * 0.25 for i in range(24)]
    ys = [exp_model(x, true_p) for x in xs]
    res = make_residual(exp_model, xs, ys)

    def jac(p):
        # d/dp of (p0 e^{p1 x} + p2 - y): [e^{p1 x}, p0 x e^{p1 x}, 1]
        J = []
        for x in xs:
            e = math.exp(p[1] * x)
            J.append([e, p[0] * x * e, 1.0])
        return J

    out_fd = levenberg_marquardt(res, [1.0, -0.2, 0.1])
    out_an = levenberg_marquardt(res, [1.0, -0.2, 0.1], jacobian=jac)
    check("analytic Jacobian == finite-diff fit",
          all(abs(out_fd["params"][i] - out_an["params"][i]) < 1e-5 for i in range(3)),
          f"{out_an['params']}")

    # ---- 7. Rosenbrock residuals driven to zero at (1,1) --------------------------------
    # r = [10(y - x^2), 1 - x]; S = Rosenbrock. Min at (1,1).
    def rosen_res(p):
        x, y = p
        return [10 * (y - x * x), 1 - x]

    out = levenberg_marquardt(rosen_res, [-1.2, 1.0], m=2, max_iter=500)
    check("Rosenbrock -> (1,1)", abs(out["params"][0] - 1) < 1e-5 and abs(out["params"][1] - 1) < 1e-5,
          f"{out['params']}")
    check("Rosenbrock cost ~ 0", out["cost"] < 1e-10, f"{out['cost']:.2e}")

    # ---- 8. recovers parameters from a noisy fit (approximately) ------------------------
    def _lcg(seed):
        st = seed & 0xFFFFFFFF

        def nxt():
            nonlocal st
            st = (1664525 * st + 1013904223) & 0xFFFFFFFF
            return (st >> 8) / (1 << 24)
        return nxt

    rng = _lcg(1)
    true_p = [2.0, -0.6, 0.5]
    xs = [i * 0.15 for i in range(40)]
    ys = [exp_model(x, true_p) + 0.02 * (rng() - 0.5) for x in xs]
    res = make_residual(exp_model, xs, ys)
    out = levenberg_marquardt(res, [1.0, -0.2, 0.2])
    check("noisy exp: params within 5% of truth",
          all(abs(out["params"][i] - true_p[i]) < 0.05 * (abs(true_p[i]) + 0.1) for i in range(3)),
          f"{[round(v,3) for v in out['params']]}")

    # ---- 9. covariance is returned and symmetric for overdetermined fits ----------------
    cov = out["covariance"]
    check("covariance returned", cov is not None and len(cov) == 3)
    if cov:
        sym = all(abs(cov[i][j] - cov[j][i]) < 1e-9 for i in range(3) for j in range(3))
        check("covariance symmetric", sym)
        check("covariance diagonal non-negative", all(cov[i][i] >= -1e-12 for i in range(3)))

    # ---- 10. already-at-optimum start converges immediately -----------------------------
    out = levenberg_marquardt(make_residual(exp_model, xs, [exp_model(x, true_p) for x in xs]),
                              list(true_p))
    check("optimal start stays optimal", out["cost"] < 1e-12, f"{out['cost']:.2e}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

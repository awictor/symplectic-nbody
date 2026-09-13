"""Tests for BFGS: quadratic exact, Rosenbrock, monotone decrease, finite-diff grad, edge cases."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bfgs import minimize, finite_diff_grad  # noqa: E402


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
    # ---- 1. quadratic: minimum at a known point -----------------------------------------
    # f = (x-3)^2 + (y+1)^2, min at (3, -1)
    f_quad = lambda v: (v[0] - 3) ** 2 + (v[1] + 1) ** 2
    g_quad = lambda v: [2 * (v[0] - 3), 2 * (v[1] + 1)]
    res = minimize(f_quad, [0.0, 0.0], grad=g_quad)
    check("quadratic minimizer ~ (3, -1)",
          abs(res["x"][0] - 3) < 1e-6 and abs(res["x"][1] + 1) < 1e-6, f"{res['x']}")
    check("quadratic value ~ 0", res["fun"] < 1e-10, f"{res['fun']:.2e}")
    check("quadratic converges fast", res["iterations"] < 10, f"{res['iterations']}")

    # ---- 2. Rosenbrock banana valley ----------------------------------------------------
    def rosen(v):
        return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2

    def rosen_grad(v):
        x, y = v
        return [-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)]
    res = minimize(rosen, [-1.2, 1.0], grad=rosen_grad, tol=1e-9, max_iter=2000)
    check("Rosenbrock minimizer ~ (1, 1)",
          abs(res["x"][0] - 1) < 1e-4 and abs(res["x"][1] - 1) < 1e-4, f"{res['x']}")
    check("Rosenbrock value ~ 0", res["fun"] < 1e-8, f"{res['fun']:.2e}")
    check("Rosenbrock gradient vanishes", res["grad_norm"] < 1e-4, f"{res['grad_norm']:.2e}")

    # ---- 3. objective decreases monotonically -------------------------------------------
    res = minimize(rosen, [-1.2, 1.0], grad=rosen_grad, tol=1e-9, max_iter=2000, track=True)
    h = res["history"]
    check("objective decreases monotonically", all(h[i + 1] <= h[i] + 1e-9 for i in range(len(h) - 1)))

    # ---- 4. higher-dimensional quadratic ------------------------------------------------
    # f = sum (x_i - i)^2, min at x_i = i
    n = 6
    def fq(v):
        return sum((v[i] - i) ** 2 for i in range(n))
    def gq(v):
        return [2 * (v[i] - i) for i in range(n)]
    res = minimize(fq, [0.0] * n, grad=gq)
    check("6-D quadratic finds x_i = i", all(abs(res["x"][i] - i) < 1e-6 for i in range(n)))

    # ---- 5. finite-difference gradient works --------------------------------------------
    res = minimize(f_quad, [0.0, 0.0])  # no analytic gradient
    check("BFGS with finite-diff gradient", abs(res["x"][0] - 3) < 1e-4 and abs(res["x"][1] + 1) < 1e-4,
          f"{res['x']}")

    # ---- 6. finite-diff gradient matches analytic ---------------------------------------
    v = [1.5, -0.7]
    fd = finite_diff_grad(rosen, v)
    an = rosen_grad(v)
    check("finite-diff gradient matches analytic", all(abs(fd[i] - an[i]) < 1e-3 for i in range(2)))

    # ---- 7. a non-polynomial function ---------------------------------------------------
    # f = sum (x_i^2) + sin-ish bowl with min at 0
    def fbowl(v):
        return sum(vi ** 2 for vi in v) + 0.1 * sum(math.cos(3 * vi) for vi in v)
    res = minimize(fbowl, [0.5, -0.3, 0.4])
    check("bowl minimizer near origin", all(abs(xi) < 0.2 for xi in res["x"]), f"{res['x']}")

    # ---- 8. already at the minimum ------------------------------------------------------
    res = minimize(f_quad, [3.0, -1.0], grad=g_quad)
    check("starting at minimum: stays there", abs(res["x"][0] - 3) < 1e-9 and abs(res["x"][1] + 1) < 1e-9)
    check("starting at minimum: quick stop", res["iterations"] <= 2)

    # ---- 9. 1-D optimization ------------------------------------------------------------
    # f = (x-5)^4, min at 5
    res = minimize(lambda v: (v[0] - 5) ** 4, [0.0], grad=lambda v: [4 * (v[0] - 5) ** 3], tol=1e-10)
    check("1-D quartic minimizer ~ 5", abs(res["x"][0] - 5) < 1e-2, f"{res['x'][0]:.4f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for barycentric interpolation: interpolates nodes, matches Lagrange, Chebyshev vs Runge."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from barycentric import (  # noqa: E402
    Barycentric,
    chebyshev_nodes,
    chebyshev_weights,
    chebyshev_interpolant,
    equispaced_interpolant,
    max_error,
    lagrange_eval,
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
    # ---- 1. interpolant passes exactly through every data point -------------------------
    rng = _lcg(2024)
    ok = True
    for _ in range(100):
        n = 2 + int(rng() * 8)
        nodes = sorted(set(rng() * 10 - 5 for _ in range(n)))
        vals = [rng() * 4 - 2 for _ in nodes]
        interp = Barycentric(nodes, vals)
        for xi, yi in zip(nodes, vals):
            if abs(interp(xi) - yi) > 1e-9:
                ok = False
    check("interpolant passes through all nodes", ok)

    # ---- 2. matches naive Lagrange evaluation everywhere --------------------------------
    rng = _lcg(77)
    maxerr = 0.0
    for _ in range(80):
        n = 3 + int(rng() * 6)
        nodes = sorted(set(rng() * 6 - 3 for _ in range(n)))
        if len(nodes) < 3:
            continue
        vals = [rng() * 4 - 2 for _ in nodes]
        interp = Barycentric(nodes, vals)
        for _ in range(10):
            xq = rng() * 6 - 3
            maxerr = max(maxerr, abs(interp(xq) - lagrange_eval(nodes, vals, xq)))
    check("barycentric == naive Lagrange", maxerr < 1e-4, f"{maxerr:.2e}")

    # ---- 3. reproduces a low-degree polynomial exactly ----------------------------------
    # p(x) = 2x^3 - x^2 + 3x - 5, sampled at 4 points -> recovered everywhere
    def p(x):
        return 2 * x ** 3 - x ** 2 + 3 * x - 5
    nodes = [-2.0, -0.5, 1.0, 3.0]
    interp = Barycentric(nodes, [p(x) for x in nodes])
    maxerr = max(abs(interp(x) - p(x)) for x in [-3, -1, 0, 0.7, 2, 4, 10])
    check("recovers cubic polynomial to machine precision", maxerr < 1e-8, f"{maxerr:.2e}")

    # ---- 4. Chebyshev weights match a direct computation --------------------------------
    n = 6
    nodes = chebyshev_nodes(n)
    direct = Barycentric(nodes, [0.0] * (n + 1))  # weights computed from product formula
    analytic = chebyshev_weights(n)
    # barycentric weights are defined up to a common scale; normalize by the first
    d = [w / direct.w[0] for w in direct.w]
    a = [w / analytic[0] for w in analytic]
    check("Chebyshev analytic weights match product formula (up to scale)",
          all(abs(d[i] - a[i]) < 1e-6 for i in range(n + 1)),
          f"{[round(x,3) for x in d[:4]]} vs {[round(x,3) for x in a[:4]]}")

    # ---- 5. Chebyshev interpolation converges geometrically for a smooth function -------
    def g(x):
        return math.exp(x) * math.sin(3 * x)
    errs = [max_error(chebyshev_interpolant(g, deg, -1, 1), g, -1, 1) for deg in [4, 8, 16, 24]]
    check("Chebyshev error decreases with degree", errs[0] > errs[1] > errs[2] > errs[3])
    check("Chebyshev degree-24 error tiny", errs[-1] < 1e-8, f"{errs[-1]:.2e}")

    # ---- 6. Runge's phenomenon: equispaced diverges, Chebyshev converges ----------------
    def runge(x):
        return 1.0 / (1 + 25 * x * x)
    eq_err = [max_error(equispaced_interpolant(runge, deg, -1, 1), runge, -1, 1)
              for deg in [8, 16, 24]]
    cheb_err = [max_error(chebyshev_interpolant(runge, deg, -1, 1), runge, -1, 1)
                for deg in [8, 16, 24]]
    check("equispaced Runge error GROWS with degree (Runge phenomenon)",
          eq_err[2] > eq_err[0], f"{eq_err[0]:.2e} -> {eq_err[2]:.2e}")
    check("Chebyshev Runge error SHRINKS with degree",
          cheb_err[2] < cheb_err[0], f"{cheb_err[0]:.2e} -> {cheb_err[2]:.2e}")
    check("Chebyshev beats equispaced on Runge at high degree", cheb_err[2] < eq_err[2])

    # ---- 7. on a wide interval / shifted -----------------------------------------------
    def h(x):
        return math.cos(x)
    interp = chebyshev_interpolant(h, 20, 0, 2 * math.pi)
    check("Chebyshev on [0, 2pi] approximates cos", max_error(interp, h, 0, 2 * math.pi) < 1e-8)

    # ---- 8. edge cases ------------------------------------------------------------------
    check("single point is constant", abs(Barycentric([3.0], [7.0])(100) - 7.0) < 1e-9)
    check("two points linear", abs(Barycentric([0.0, 2.0], [1.0, 5.0])(1.0) - 3.0) < 1e-12)
    try:
        Barycentric([1.0, 1.0], [2.0, 3.0])  # duplicate node
        check("duplicate node raises", False)
    except ValueError:
        check("duplicate node raises", True)
    try:
        Barycentric([1.0, 2.0], [3.0])  # mismatch
        check("length mismatch raises", False)
    except ValueError:
        check("length mismatch raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

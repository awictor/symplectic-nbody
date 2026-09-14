"""Tests for Richardson extrapolation: derivatives to machine precision, limits, order improvement."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from richardson_extrapolation import (  # noqa: E402
    richardson_tableau, richardson_extrapolate, derivative, derivative_tableau,
    second_derivative, limit, convergence_order, _central_difference,
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
    # ---- 1. derivative to near machine precision ----------------------------------------
    check("d/dx sin at 1 = cos(1)", abs(derivative(math.sin, 1) - math.cos(1)) < 1e-12,
          f"{derivative(math.sin, 1)}")
    check("d/dx exp at 0 = 1", abs(derivative(math.exp, 0) - 1.0) < 1e-12)
    check("d/dx exp at 2 = e^2", abs(derivative(math.exp, 2) - math.exp(2)) < 1e-10)
    check("d/dx cos at 0.5", abs(derivative(math.cos, 0.5) - (-math.sin(0.5))) < 1e-12)

    # ---- 2. polynomial derivative exact -------------------------------------------------
    p = lambda x: 3 * x ** 3 - 2 * x ** 2 + x - 5
    dp = lambda x: 9 * x ** 2 - 4 * x + 1
    for x in [-1, 0, 1, 2.5]:
        check(f"poly derivative at {x}", abs(derivative(p, x) - dp(x)) < 1e-8, f"{derivative(p,x)} vs {dp(x)}")

    # ---- 3. extrapolation beats the raw central difference ------------------------------
    x = 1.0
    raw_err = abs(_central_difference(math.sin, x, 0.5) - math.cos(x))
    ext_err = abs(derivative(math.sin, x) - math.cos(x))
    check("extrapolation >> raw central difference", ext_err < raw_err / 1e6,
          f"raw {raw_err:.2e} ext {ext_err:.2e}")

    # ---- 4. second derivative -----------------------------------------------------------
    check("d2/dx2 sin at 1 = -sin(1)", abs(second_derivative(math.sin, 1) - (-math.sin(1))) < 1e-7,
          f"{second_derivative(math.sin, 1)}")
    check("d2/dx2 x^4 at 2 = 48", abs(second_derivative(lambda x: x ** 4, 2) - 48) < 1e-4)

    # ---- 5. limit extrapolation: (1+h)^(1/h) -> e ---------------------------------------
    e_est = limit(lambda h: (1 + h) ** (1 / h), h0=1.0, p=1, t=2, levels=10)
    check("(1+h)^(1/h) -> e", abs(e_est - math.e) < 1e-7, f"{e_est}")

    # ---- 6. limit: sin(h)/h -> 1 --------------------------------------------------------
    s_est = limit(lambda h: math.sin(h) / h, h0=0.5, p=2, t=2, levels=8)
    check("sin(h)/h -> 1", abs(s_est - 1.0) < 1e-12, f"{s_est}")

    # ---- 7. limit: (cos h - 1)/h^2 -> -1/2 ----------------------------------------------
    c_est = limit(lambda h: (math.cos(h) - 1) / (h * h), h0=0.5, p=2, t=2, levels=8)
    check("(cos h - 1)/h^2 -> -1/2", abs(c_est - (-0.5)) < 1e-10, f"{c_est}")

    # ---- 8. tableau reproduces Romberg on the trapezoid ladder --------------------------
    # A(h) = trapezoid estimate of int_0^1 e^x dx with step h; Richardson (p=2) -> Romberg
    def trap(h):
        n = int(round(1.0 / h))
        h = 1.0 / n
        total = 0.5 * (math.exp(0) + math.exp(1))
        for k in range(1, n):
            total += math.exp(k * h)
        return h * total
    # start with h = 1/2, refine by 2
    romberg_est = richardson_extrapolate(lambda h: trap(h), 0.5, p=2, t=2, levels=6)
    check("Richardson trapezoid -> integral e^x", abs(romberg_est - (math.e - 1)) < 1e-10,
          f"{romberg_est}")

    # ---- 9. convergence order of the derivative tableau diagonal is ~2,4,6 --------------
    orders = convergence_order(lambda h: _central_difference(math.sin, 1, h), 0.5, 2, 2, 6)
    # first order should be near 4 (central diff h^2 -> extrapolated h^4), rising
    check("convergence order improves (>=3 in early columns)", orders[0] >= 3.0, f"{orders}")

    # ---- 10. general tableau with p=1 ---------------------------------------------------
    # A(h) = 2 + 3h + h^2 -> limit 2 as h->0
    A = lambda h: 2 + 3 * h + h * h
    est = richardson_extrapolate(A, 1.0, p=1, t=2, levels=5)
    check("linear-order extrapolation to 2", abs(est - 2.0) < 1e-9, f"{est}")

    # ---- 11. tableau shape --------------------------------------------------------------
    T = derivative_tableau(math.sin, 1, levels=5)
    check("tableau is lower-triangular with 5 rows", len(T) == 5 and len(T[4]) == 5)

    # ---- 12. best estimate is the corner ------------------------------------------------
    T = derivative_tableau(math.exp, 0, levels=6)
    check("corner is best derivative estimate", abs(T[-1][-1] - 1.0) < 1e-12)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

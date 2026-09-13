"""Tests for Pade approximants: series matching, geometric pole exact, beats Taylor, edge cases."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pade import (  # noqa: E402
    pade,
    evaluate,
    pade_series,
    taylor_coeffs_from_function,
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


def _taylor_eval(coeffs, x):
    s = 0.0
    for c in reversed(coeffs):
        s = s * x + c
    return s


def main():
    # ---- 1. Pade series matches input Taylor coefficients through order m+n -------------
    # exp: c_k = 1/k!
    exp_c = [1.0 / math.factorial(k) for k in range(8)]
    for (m, n) in [(2, 2), (3, 1), (1, 3), (4, 0), (0, 4)]:
        P, Q = pade(exp_c, m, n)
        series = pade_series(P, Q, m + n)
        ok = all(abs(series[k] - exp_c[k]) < 1e-9 for k in range(m + n + 1))
        check(f"[{m}/{n}] Pade series matches exp Taylor through order {m+n}", ok)

    # ---- 2. [m/0] reduces to the Taylor polynomial --------------------------------------
    P, Q = pade(exp_c, 3, 0)
    check("[3/0] denominator is 1", Q == [1.0])
    check("[3/0] numerator is the Taylor poly", all(abs(P[k] - exp_c[k]) < 1e-12 for k in range(4)))

    # ---- 3. [1/1] of 1/(1-x) is exact (recovers the pole) -------------------------------
    # 1/(1-x) = 1 + x + x^2 + ...  -> c_k = 1
    geo_c = [1.0] * 4
    P, Q = pade(geo_c, 1, 1)
    # should be 1 / (1 - x): P = [1], Q = [1, -1]
    check("[1/1] of geometric: exact at several points",
          all(abs(evaluate(P, Q, x) - 1 / (1 - x)) < 1e-9 for x in [0.3, 0.5, 0.9, -0.4, 2.0]),
          f"P={[round(p,3) for p in P]} Q={[round(q,3) for q in Q]}")
    check("[1/1] denominator recovers the pole at x=1", abs(Q[1] + 1) < 1e-9, f"{Q}")

    # ---- 4. Pade beats Taylor for exp over a wide interval ------------------------------
    exp_c = [1.0 / math.factorial(k) for k in range(9)]
    P, Q = pade(exp_c, 4, 4)
    taylor8 = exp_c[:9]
    def err_at(x):
        true = math.exp(x)
        return abs(evaluate(P, Q, x) - true), abs(_taylor_eval(taylor8, x) - true)
    # at x = 3 the [4/4] Pade should be far more accurate than the degree-8 Taylor
    pe, te = err_at(3.0)
    check("Pade [4/4] beats Taylor(8) for exp at x=3", pe < te / 3, f"Pade {pe:.2e} vs Taylor {te:.2e}")
    pe2, te2 = err_at(-3.0)
    check("Pade [4/4] beats Taylor(8) for exp at x=-3", pe2 < te2, f"Pade {pe2:.2e} vs Taylor {te2:.2e}")

    # ---- 5. log(1+x): Pade extends accuracy past the Taylor radius (|x|<1) --------------
    # log(1+x) = x - x^2/2 + x^3/3 - ..., c_0 = 0, c_k = (-1)^(k+1)/k
    log_c = [0.0] + [(-1) ** (k + 1) / k for k in range(1, 9)]
    P, Q = pade(log_c, 4, 4)
    taylor = log_c
    # at x = 2 (outside the Taylor radius 1) Pade should still be reasonable
    true = math.log(3)
    pe = abs(evaluate(P, Q, 2.0) - true)
    te = abs(_taylor_eval(taylor, 2.0) - true)
    check("Pade log(1+x) accurate past Taylor radius (x=2)", pe < te, f"Pade {pe:.3f} vs Taylor {te:.3f}")
    check("Pade log(1+x) small error at x=2", pe < 0.05, f"{pe:.4f}")

    # ---- 6. arctan Pade -----------------------------------------------------------------
    # arctan: c_k nonzero for odd k = (-1)^((k-1)/2)/k
    atan_c = [0.0] * 9
    for k in range(1, 9, 2):
        atan_c[k] = (-1) ** ((k - 1) // 2) / k
    P, Q = pade(atan_c, 4, 4)
    pe = abs(evaluate(P, Q, 1.0) - math.atan(1.0))
    check("Pade arctan accurate at x=1", pe < 5e-3, f"{pe:.2e}")

    # ---- 7. build from a function via finite-difference Taylor coeffs -------------------
    c = taylor_coeffs_from_function(math.exp, 4, x0=0.0, h=1e-2)
    # should approximate 1/k!
    check("finite-diff Taylor coeffs approximate exp", all(abs(c[k] - 1 / math.factorial(k)) < 1e-2
                                                           for k in range(5)), f"{[round(x,4) for x in c]}")

    # ---- 8. edge cases ------------------------------------------------------------------
    try:
        pade([1.0, 1.0], 2, 2)  # not enough coeffs
        check("insufficient coeffs raises", False)
    except ValueError:
        check("insufficient coeffs raises", True)
    # [0/0] is just the constant c_0
    P, Q = pade([5.0], 0, 0)
    check("[0/0] is the constant term", P == [5.0] and Q == [1.0])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

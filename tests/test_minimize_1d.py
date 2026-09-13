"""Tests for minimize_1d: known minima, Brent faster than golden, bracketing, edge cases."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minimize_1d import (golden_section, brent, bracket_minimum, minimize, _GOLDEN)  # noqa: E402


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
    # ---- 1. known minima, both methods -------------------------------------------------
    cases = [
        ("parabola min at 3", lambda x: (x - 3) ** 2 + 1, -10, 10, 3.0),
        ("x^4 min at 0", lambda x: x ** 4, -5, 5, 0.0),
        ("cos min at pi", math.cos, 0, 2 * math.pi, math.pi),
        ("neg gaussian min at 2", lambda x: -math.exp(-(x - 2) ** 2), -5, 8, 2.0),
        ("shifted abs-smooth", lambda x: (x + 1.5) ** 2 + 0.3 * math.sin(x), -6, 6, None),
    ]
    for name, f, a, b, xmin in cases:
        rg = golden_section(f, a, b, tol=1e-9)
        rb = brent(f, a, b, tol=1e-10)
        if xmin is not None:
            check(f"golden: {name}", abs(rg.x - xmin) < 1e-5, f"got {rg.x}")
            check(f"brent: {name}", abs(rb.x - xmin) < 1e-6, f"got {rb.x}")
        else:
            # both should agree on the same minimizer
            check(f"golden and brent agree: {name}", abs(rg.x - rb.x) < 1e-4,
                  f"gs {rg.x} brent {rb.x}")
            # and it is a genuine local min (lower than neighbours)
            xm = rb.x
            check(f"is a local min: {name}", f(xm) <= f(xm - 1e-4) and f(xm) <= f(xm + 1e-4))

    # ---- 2. Brent uses fewer evaluations than golden section on smooth functions ------
    f = lambda x: (x - 3.7) ** 2 + 2
    rg = golden_section(f, -20, 20, tol=1e-10)
    rb = brent(f, -20, 20, tol=1e-10)
    check("both find the same minimizer", abs(rg.x - rb.x) < 1e-6)
    check("Brent uses fewer evaluations than golden", rb.evaluations < rg.evaluations,
          f"brent {rb.evaluations} vs golden {rg.evaluations}")

    # ---- 3. automatic bracketing ------------------------------------------------------
    for target in (7.0, -4.0, 0.0):
        f = lambda x, t=target: (x - t) ** 2
        a, b, c = bracket_minimum(f, x0=0.0, step=1.0)
        check(f"bracket brackets the min at {target}", a <= target <= c and f(b) <= f(a)
              and f(b) <= f(c), f"bracket ({a},{b},{c})")
    # bracket works when the minimum is to the left of x0 too
    f = lambda x: (x + 20) ** 2
    a, b, c = bracket_minimum(f, x0=5.0, step=1.0)
    check("bracket finds a minimum to the left", f(b) <= f(a) and f(b) <= f(c))

    # ---- 4. minimize() auto-brackets and solves ---------------------------------------
    r = minimize(lambda x: (x - 12.5) ** 2 + 1, x0=0.0, step=1.0)
    check("minimize auto-brackets and finds the min", abs(r.x - 12.5) < 1e-6, f"{r.x}")
    r = minimize(lambda x: (x - 12.5) ** 2 + 1, a=0, b=30, method="golden")
    check("minimize with explicit bracket + golden", abs(r.x - 12.5) < 1e-5)

    # ---- 5. golden section shrinks by the golden ratio --------------------------------
    # track interval widths on a parabola
    f = lambda x: x * x
    a, b = -1.0, 1.0
    x1 = a + (1 - _GOLDEN) * (b - a)
    x2 = a + _GOLDEN * (b - a)
    f1, f2 = f(x1), f(x2)
    widths = [b - a]
    for _ in range(5):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + (1 - _GOLDEN) * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + _GOLDEN * (b - a)
            f2 = f(x2)
        widths.append(b - a)
    ratios = [widths[i + 1] / widths[i] for i in range(len(widths) - 1)]
    check("golden section shrinks by ~0.618 each step",
          all(abs(r - _GOLDEN) < 1e-9 for r in ratios), f"{ratios}")

    # ---- 6. minimum at a bracket endpoint ---------------------------------------------
    # monotone on [0, 5], min at x=0
    f = lambda x: (x + 3) ** 2
    r = brent(f, 0, 5, tol=1e-9)
    check("minimum at left endpoint found", abs(r.x - 0.0) < 1e-4, f"{r.x}")

    # ---- 7. narrow valley -------------------------------------------------------------
    f = lambda x: 1000 * (x - 1.23456) ** 2
    r = brent(f, -10, 10, tol=1e-12)
    check("narrow steep valley located precisely", abs(r.x - 1.23456) < 1e-6, f"{r.x}")

    # ---- 8. multimodal: only guarantees a LOCAL min within the bracket ----------------
    # two wells; bracket around the right one
    f = lambda x: math.sin(x) + 0.1 * x
    r = brent(f, 3, 6, tol=1e-9)   # a local min near 4.7
    check("finds a local min of a multimodal function", f(r.x) <= f(r.x - 1e-3)
          and f(r.x) <= f(r.x + 1e-3))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

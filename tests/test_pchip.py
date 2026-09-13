"""Tests for PCHIP: passes through knots, monotone on monotone data, no overshoot, C1."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pchip import PCHIP  # noqa: E402


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


def _fine(x0, x1, n=200):
    return [x0 + (x1 - x0) * k / n for k in range(n + 1)]


def main():
    # ---- 1. passes exactly through every knot -------------------------------------------
    rng = _lcg(2024)
    ok = True
    for _ in range(100):
        n = 3 + int(rng() * 6)
        x = [0.0]
        for _ in range(n - 1):
            x.append(x[-1] + 0.2 + rng() * 2)
        y = [rng() * 10 - 5 for _ in range(n)]
        p = PCHIP(x, y)
        for xi, yi in zip(x, y):
            if abs(p(xi) - yi) > 1e-9:
                ok = False
    check("interpolant passes through all knots", ok)

    # ---- 2. monotone-increasing data -> monotone-increasing interpolant -----------------
    rng = _lcg(77)
    ok = True
    for _ in range(100):
        n = 4 + int(rng() * 6)
        x = [float(i) for i in range(n)]
        # strictly increasing y
        y = [0.0]
        for _ in range(n - 1):
            y.append(y[-1] + 0.01 + rng() * 3)
        p = PCHIP(x, y)
        xs = _fine(x[0], x[-1], 300)
        ys = p.evaluate(xs)
        if any(ys[k + 1] < ys[k] - 1e-9 for k in range(len(ys) - 1)):
            ok = False
    check("monotone-increasing data -> monotone interpolant", ok)

    # ---- 3. no overshoot: interpolant stays within local data range ---------------------
    rng = _lcg(7)
    ok = True
    for _ in range(100):
        n = 4 + int(rng() * 5)
        x = [float(i) for i in range(n)]
        y = [rng() * 10 for _ in range(n)]
        p = PCHIP(x, y)
        for i in range(n - 1):
            lo = min(y[i], y[i + 1])
            hi = max(y[i], y[i + 1])
            for xq in _fine(x[i], x[i + 1], 30):
                v = p(xq)
                if v < lo - 1e-9 or v > hi + 1e-9:
                    ok = False
    check("no overshoot: interpolant within local [y_i, y_{i+1}]", ok)

    # ---- 4. classic step data: PCHIP monotone where a natural spline overshoots ---------
    # data that rises then plateaus -- the textbook overshoot trap
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 0, 0, 1, 1, 1]
    p = PCHIP(x, y)
    xs = _fine(0, 5, 500)
    ys = p.evaluate(xs)
    check("step data: interpolant stays in [0,1]", all(-1e-9 <= v <= 1 + 1e-9 for v in ys))
    check("step data: monotone non-decreasing", all(ys[k + 1] >= ys[k] - 1e-9
                                                     for k in range(len(ys) - 1)))
    # local extrema (flat regions) get zero slope
    check("plateau start has zero slope", abs(p.slopes()[1]) < 1e-12)

    # ---- 5. reproduces a straight line exactly ------------------------------------------
    x = [0, 1, 3, 6, 10]
    y = [2 + 3 * xi for xi in x]  # line slope 3
    p = PCHIP(x, y)
    maxerr = max(abs(p(xq) - (2 + 3 * xq)) for xq in _fine(0, 10, 100))
    check("reproduces a straight line exactly", maxerr < 1e-9, f"{maxerr:.2e}")
    check("line derivative is the slope", abs(p.derivative(5) - 3) < 1e-9)

    # ---- 6. C1 continuity: derivative continuous at interior knots ----------------------
    rng = _lcg(321)
    ok = True
    for _ in range(50):
        n = 4 + int(rng() * 4)
        x = [float(i) for i in range(n)]
        y = [rng() * 10 - 5 for _ in range(n)]
        p = PCHIP(x, y)
        for i in range(1, n - 1):
            left = p.derivative(x[i] - 1e-7)
            right = p.derivative(x[i] + 1e-7)
            if abs(left - right) > 1e-3:
                ok = False
    check("C1: derivative continuous at knots", ok)

    # ---- 7. decreasing data works too ---------------------------------------------------
    x = [0, 1, 2, 3, 4]
    y = [10, 7, 3, 1, 0]  # strictly decreasing
    p = PCHIP(x, y)
    xs = _fine(0, 4, 200)
    ys = p.evaluate(xs)
    check("decreasing data -> monotone decreasing", all(ys[k + 1] <= ys[k] + 1e-9
                                                        for k in range(len(ys) - 1)))

    # ---- 8. edge cases ------------------------------------------------------------------
    check("two points = straight line", abs(PCHIP([0, 2], [1, 5])(1) - 3) < 1e-9)
    try:
        PCHIP([0, 0], [1, 2])  # non-increasing x
        check("non-increasing x raises", False)
    except ValueError:
        check("non-increasing x raises", True)
    try:
        PCHIP([0], [1])  # single point
        check("single point raises", False)
    except ValueError:
        check("single point raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

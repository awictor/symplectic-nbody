"""Validate Theil-Sen / Siegel: exact recovery, outlier resistance vs OLS, breakdown, hand-checkable cases."""

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import theil_sen as ts


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def uniform(self, a, b):
        return a + (b - a) * self.u()


def main():
    print("Theil-Sen tests")

    # --- exact recovery on a clean line ---
    xs = [float(i) for i in range(20)]
    m_true, b_true = 2.5, -1.3
    ys = [m_true * x + b_true for x in xs]
    m, b = ts.theil_sen(xs, ys)
    check("clean line slope exact", abs(m - m_true) < 1e-9)
    check("clean line intercept exact", abs(b - b_true) < 1e-9)

    # --- hand-checkable pairwise-slope median ---
    # points (0,0),(1,2),(2,4),(3,100): slopes between pairs; median should be 2 (the clean trend),
    # NOT dragged by the outlier at (3,100).
    hx = [0.0, 1.0, 2.0, 3.0]
    hy = [0.0, 2.0, 4.0, 100.0]
    mh, bh = ts.theil_sen(hx, hy)
    # pairwise slopes: (2,2,2, ... to the outlier: 100,98,96) -> sorted median of 6 slopes
    slopes = sorted([(hy[j] - hy[i]) / (hx[j] - hx[i]) for i in range(4) for j in range(i + 1, 4)])
    med = 0.5 * (slopes[2] + slopes[3])
    check("hand-checkable pairwise-slope median", abs(mh - med) < 1e-12)

    # --- OUTLIER RESISTANCE: 20% of points corrupted, Theil-Sen holds, OLS breaks ---
    n = 50
    rng = _R(7)
    xs = [float(i) for i in range(n)]
    ys = [3.0 * x + 5.0 + rng.uniform(-0.5, 0.5) for x in xs]
    # corrupt 20% with huge outliers
    n_out = n // 5
    for k in range(n_out):
        idx = (k * 7) % n
        ys[idx] += 500.0
    m_ts, b_ts = ts.theil_sen(xs, ys)
    m_ols, b_ols = ts.ols(xs, ys)
    check(f"Theil-Sen slope robust to 20% outliers (got {m_ts:.3f}, true 3.0)", abs(m_ts - 3.0) < 0.3)
    check(f"OLS slope corrupted by outliers (got {m_ols:.3f})", abs(m_ols - 3.0) > 1.0)
    check("Theil-Sen far closer to truth than OLS", abs(m_ts - 3.0) < abs(m_ols - 3.0))

    # --- Siegel repeated median survives ~40% outliers where Theil-Sen weakens ---
    n = 40
    xs = [float(i) for i in range(n)]
    ys = [1.5 * x + 2.0 for x in xs]
    n_out = int(n * 0.40)
    for k in range(n_out):
        idx = (k * 3) % n
        ys[idx] = 1000.0  # gross outliers
    m_sg, b_sg = ts.siegel_repeated_median(xs, ys)
    check(f"Siegel survives 40% outliers (slope {m_sg:.3f}, true 1.5)", abs(m_sg - 1.5) < 0.2)

    # --- horizontal line ---
    xs = [float(i) for i in range(10)]
    ys = [7.0] * 10
    m, b = ts.theil_sen(xs, ys)
    check("horizontal line: zero slope", abs(m) < 1e-12 and abs(b - 7.0) < 1e-12)

    # --- steep line ---
    xs = [float(i) for i in range(10)]
    ys = [1000.0 * x for x in xs]
    m, b = ts.theil_sen(xs, ys)
    check("steep line recovered", abs(m - 1000.0) < 1e-6)

    # --- prediction consistency ---
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 3.0, 5.0, 7.0, 9.0]  # slope 2, intercept 1
    model = ts.theil_sen(xs, ys)
    check("predict on the line", abs(ts.predict(model, 10.0) - 21.0) < 1e-9)

    # --- confidence interval brackets the true slope ---
    n = 40
    rng = _R(99)
    xs = [float(i) for i in range(n)]
    ys = [2.0 * x + 1.0 + rng.uniform(-2, 2) for x in xs]
    lo, hi = ts.slope_confidence_interval(xs, ys, alpha=0.05)
    check(f"95% CI brackets true slope 2.0 ([{lo:.3f},{hi:.3f}])", lo <= 2.0 <= hi)
    check("CI is an ordered interval", lo <= hi)

    # --- deterministic ---
    a = ts.theil_sen(xs, ys)
    c = ts.theil_sen(xs, ys)
    check("deterministic", a == c)

    # --- inverse-normal helper sanity ---
    check("inv_norm(0.5)=0", abs(ts._inv_norm(0.5)) < 1e-6)
    check("inv_norm(0.975)~1.96", abs(ts._inv_norm(0.975) - 1.959964) < 1e-3)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

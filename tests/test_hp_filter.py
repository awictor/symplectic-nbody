"""Tests for HP filter: trend+cycle reconstruct, lambda limits, banded==dense, smoothness, noise."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hp_filter import (  # noqa: E402
    hp_filter,
    hp_dense,
    second_difference_norm,
    objective,
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
    # ---- 1. trend + cycle reconstructs the data -----------------------------------------
    rng = _lcg(2024)
    ok = True
    for _ in range(30):
        n = 5 + int(rng() * 30)
        y = [rng() * 10 for _ in range(n)]
        tau, cyc = hp_filter(y, lam=100)
        if any(abs(tau[i] + cyc[i] - y[i]) > 1e-9 for i in range(n)):
            ok = False
    check("trend + cycle == data", ok)

    # ---- 2. lambda = 0 -> trend is the data, zero cycle ---------------------------------
    y = [1.0, 4.0, 2.0, 8.0, 3.0]
    tau, cyc = hp_filter(y, 0.0)
    check("lambda=0: trend equals data", tau == y)
    check("lambda=0: zero cycle", all(abs(c) < 1e-12 for c in cyc))

    # ---- 3. huge lambda -> trend is the least-squares straight line ---------------------
    rng = _lcg(77)
    n = 20
    # y = 2 + 0.5 t + noise; huge lambda should recover the line 2 + 0.5 t
    y = [2 + 0.5 * t + (rng() - 0.5) * 2 for t in range(n)]
    tau, _ = hp_filter(y, lam=1e8)
    # least-squares line fit of y
    mx = sum(range(n)) / n
    my = sum(y) / n
    sxx = sum((t - mx) ** 2 for t in range(n))
    sxy = sum((t - mx) * (y[t] - my) for t in range(n))
    slope = sxy / sxx
    inter = my - slope * mx
    line = [inter + slope * t for t in range(n)]
    check("huge lambda -> least-squares line",
          max(abs(tau[t] - line[t]) for t in range(n)) < 1e-3,
          f"{max(abs(tau[t] - line[t]) for t in range(n)):.2e}")

    # ---- 4. banded solve matches dense reference ----------------------------------------
    rng = _lcg(7)
    maxerr = 0.0
    for _ in range(30):
        n = 5 + int(rng() * 25)
        y = [rng() * 10 for _ in range(n)]
        lam = 0.5 + rng() * 100
        tau_b, _ = hp_filter(y, lam)
        tau_d = hp_dense(y, lam)
        maxerr = max(maxerr, max(abs(tau_b[i] - tau_d[i]) for i in range(n)))
    check("banded solve == dense reference", maxerr < 1e-8, f"{maxerr:.2e}")

    # ---- 5. trend minimizes the HP objective --------------------------------------------
    rng = _lcg(555)
    ok = True
    for _ in range(20):
        n = 8 + int(rng() * 10)
        y = [rng() * 10 for _ in range(n)]
        lam = 1.0 + rng() * 20
        tau, _ = hp_filter(y, lam)
        base = objective(y, tau, lam)
        # perturb the trend; objective must not decrease
        for _ in range(10):
            pert = list(tau)
            i = int(rng() * n)
            pert[i] += (rng() - 0.5) * 0.5
            if objective(y, pert, lam) < base - 1e-6:
                ok = False
    check("trend minimizes HP objective (no perturbation lowers it)", ok)

    # ---- 6. larger lambda -> smoother trend ---------------------------------------------
    rng = _lcg(11)
    y = [rng() * 10 for _ in range(40)]
    s_small = second_difference_norm(hp_filter(y, 1.0)[0])
    s_large = second_difference_norm(hp_filter(y, 1000.0)[0])
    check("larger lambda gives smoother trend", s_large < s_small, f"{s_small:.3f} -> {s_large:.3f}")

    # ---- 7. noise pushed into the cycle -------------------------------------------------
    rng = _lcg(321)
    n = 60
    true_trend = [t * 0.3 for t in range(n)]  # smooth linear trend
    y = [true_trend[t] + (rng() - 0.5) * 3 for t in range(n)]
    tau, cyc = hp_filter(y, lam=1000)
    # trend should be close to the true smooth trend; cycle carries the noise
    trend_err = sum((tau[t] - true_trend[t]) ** 2 for t in range(n)) / n
    raw_err = sum((y[t] - true_trend[t]) ** 2 for t in range(n)) / n
    check("trend closer to true than noisy data", trend_err < raw_err, f"{trend_err:.3f} vs {raw_err:.3f}")
    check("cycle mean near zero", abs(sum(cyc) / n) < 0.5, f"{sum(cyc)/n:.4f}")

    # ---- 8. edge cases ------------------------------------------------------------------
    check("empty series", hp_filter([], 100) == ([], []))
    check("two-point series returned as-is", hp_filter([1.0, 2.0], 100)[0] == [1.0, 2.0])
    tau, cyc = hp_filter([5.0], 100)
    check("single point", tau == [5.0] and cyc == [0.0])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for exponential smoothing: SES on constant, Holt on trend, HW on seasonal, alpha reactivity."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from holt_winters import (  # noqa: E402
    ses,
    holt,
    holt_winters,
    one_step_errors,
    rmse,
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
    # ---- 1. SES tracks a noisy constant to its mean, forecasts flat ---------------------
    rng = _lcg(2024)
    y = [10 + (rng() - 0.5) * 2 for _ in range(100)]
    levels, forecast = ses(y, alpha=0.2)
    check("SES level converges near the mean 10", abs(levels[-1] - 10) < 1.0, f"{levels[-1]:.3f}")
    fc = forecast(5)
    check("SES forecast is flat", all(abs(f - fc[0]) < 1e-12 for f in fc))
    check("SES forecast near the mean", abs(fc[0] - 10) < 1.0)

    # ---- 2. Holt recovers a linear trend ------------------------------------------------
    y = [3 + 2 * t for t in range(30)]  # exact line, slope 2
    levels, trends, forecast = holt(y, alpha=0.5, beta=0.5)
    check("Holt trend ~ slope 2", abs(trends[-1] - 2) < 0.05, f"{trends[-1]:.4f}")
    fc = forecast(5)
    # forecast should continue the line: y[29]=61, next values 63,65,67,69,71
    expected = [3 + 2 * (30 + i) for i in range(5)]
    check("Holt forecast extrapolates the line", all(abs(fc[i] - expected[i]) < 0.5 for i in range(5)),
          f"{[round(f,1) for f in fc]} vs {expected}")

    # ---- 3. Holt-Winters reproduces a trend + seasonal series ---------------------------
    m = 12
    def hw_signal(t):
        return 50 + 0.5 * t + 10 * math.sin(2 * math.pi * (t % m) / m)
    y = [hw_signal(t) for t in range(48)]
    levels, trends, seasonals, forecast = holt_winters(y, 0.3, 0.1, 0.3, period=m)
    fc = forecast(m)
    expected = [hw_signal(48 + i) for i in range(m)]
    err = max(abs(fc[i] - expected[i]) for i in range(m))
    check("Holt-Winters forecast reinstates trend+seasonal", err < 5.0, f"max err {err:.3f}")
    # the seasonal component should have the sinusoidal shape (peak and trough)
    check("HW seasonal has spread", max(seasonals) - min(seasonals) > 10)

    # ---- 4. higher alpha reacts faster to a level shift ---------------------------------
    # step: 0 for 50 steps then jump to 10
    y = [0.0] * 50 + [10.0] * 20
    lv_slow, _ = ses(y, alpha=0.1)
    lv_fast, _ = ses(y, alpha=0.6)
    # a few steps after the jump, the fast filter should be closer to 10
    idx = 55
    check("higher alpha reacts faster to level shift",
          abs(lv_fast[idx] - 10) < abs(lv_slow[idx] - 10),
          f"fast {lv_fast[idx]:.2f} slow {lv_slow[idx]:.2f}")

    # ---- 5. one-step error small for a well-specified model -----------------------------
    y = [5 + 3 * t for t in range(40)]
    levels, trends, forecast = holt(y, 0.5, 0.5)
    # one-step forecast from Holt = level + trend at previous step
    preds = [levels[t - 1] + trends[t - 1] for t in range(1, len(y))]
    errs = [y[t] - preds[t - 1] for t in range(1, len(y))]
    check("Holt one-step error small on a line", rmse(errs) < 0.5, f"rmse {rmse(errs):.4f}")

    # ---- 6. SES one-step errors on white noise ~ noise level ----------------------------
    rng = _lcg(77)
    y = [(rng() - 0.5) * 2 for _ in range(200)]
    levels, _ = ses(y, 0.3)
    errs = one_step_errors(y, levels)
    check("SES one-step errors computed", len(errs) == len(y) - 1)

    # ---- 7. parameter validation --------------------------------------------------------
    for bad in [ses, lambda y, a: holt(y, a, 0.5)]:
        try:
            bad([1.0, 2.0, 3.0], 1.5)  # alpha out of range
            check("out-of-range alpha raises", False)
            break
        except ValueError:
            pass
    else:
        check("out-of-range alpha raises", True)

    try:
        holt_winters([1.0] * 5, 0.3, 0.1, 0.3, period=12)  # too few points
        check("insufficient HW data raises", False)
    except ValueError:
        check("insufficient HW data raises", True)

    # ---- 8. forecast horizon length ------------------------------------------------------
    _, fc = ses([1.0, 2.0, 3.0], 0.5)
    check("SES forecast returns h values", len(fc(7)) == 7)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for isotonic regression (PAVA): monotone, matches min-max formula, block KKT, calibration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isotonic import (  # noqa: E402
    isotonic_regression,
    blocks,
    IsotonicModel,
    sse,
    brute_isotonic,
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
        return state >> 8

    return nxt


def _is_monotone(f, tol=1e-9):
    return all(f[i] <= f[i + 1] + tol for i in range(len(f) - 1))


def main():
    # ---- 1. fit is monotone -------------------------------------------------------------
    rng = _lcg(2024)
    ok = True
    for _ in range(300):
        n = 1 + rng() % 15
        y = [(rng() % 100) - 50 for _ in range(n)]
        f = isotonic_regression(y)
        if not _is_monotone(f):
            ok = False
    check("fit is always non-decreasing", ok)

    # ---- 2. matches the exact min-max formula -------------------------------------------
    rng = _lcg(77)
    maxerr = 0.0
    for _ in range(200):
        n = 1 + rng() % 10
        y = [float((rng() % 40) - 20) for _ in range(n)]
        f = isotonic_regression(y)
        b = brute_isotonic(y)
        maxerr = max(maxerr, max((abs(a - c) for a, c in zip(f, b)), default=0.0))
    check("PAVA == exact min-max formula (200 cases)", maxerr < 1e-9, f"max err {maxerr:.2e}")

    # ---- 3. weighted case matches formula -----------------------------------------------
    rng = _lcg(7)
    maxerr = 0.0
    for _ in range(150):
        n = 1 + rng() % 9
        y = [float((rng() % 40) - 20) for _ in range(n)]
        w = [1 + rng() % 5 for _ in range(n)]
        f = isotonic_regression(y, weights=w)
        b = brute_isotonic(y, weights=w)
        maxerr = max(maxerr, max((abs(a - c) for a, c in zip(f, b)), default=0.0))
    check("weighted PAVA == weighted formula", maxerr < 1e-9, f"max err {maxerr:.2e}")

    # ---- 4. block values are the weighted mean of their members (KKT) -------------------
    rng = _lcg(555)
    ok = True
    for _ in range(150):
        n = 1 + rng() % 12
        y = [float((rng() % 40) - 20) for _ in range(n)]
        w = [1 + rng() % 4 for _ in range(n)]
        f = isotonic_regression(y, weights=w)
        for (start, length, val) in blocks(y, weights=w):
            sw = sum(w[start:start + length])
            wm = sum(w[start + t] * y[start + t] for t in range(length)) / sw
            if abs(wm - val) > 1e-9:
                ok = False
    check("each block value is the weighted mean of its members", ok)

    # ---- 5. optimality: PAVA SSE <= any monotone perturbation ---------------------------
    rng = _lcg(11)
    ok = True
    for _ in range(100):
        n = 2 + rng() % 8
        y = [float((rng() % 30) - 15) for _ in range(n)]
        f = isotonic_regression(y)
        base = sse(y, f)
        # perturb into another monotone vector and check SSE does not decrease
        for _ in range(5):
            g = list(f)
            i = rng() % n
            g[i] += (rng() / (1 << 24) - 0.5) * 2
            # re-monotonize
            g = isotonic_regression(g) if not _is_monotone(g) else g
            # move g toward being a valid monotone competitor by re-projecting the data-free vector
            if _is_monotone(g) and sse(y, g) < base - 1e-9:
                ok = False
    check("PAVA fit is optimal (no monotone vector has lower SSE)", ok)

    # ---- 6. non-increasing via flip -----------------------------------------------------
    y = [5, 3, 4, 1, 2]
    f = isotonic_regression(y, increasing=False)
    check("non-increasing fit is non-increasing", all(f[i] >= f[i + 1] - 1e-9 for i in range(len(f) - 1)))

    # ---- 7. already-monotone data unchanged ---------------------------------------------
    mono = [1.0, 2.0, 2.0, 3.5, 9.0]
    check("already non-decreasing data unchanged", isotonic_regression(mono) == mono)

    # ---- 8. recovers a monotone signal from noise ---------------------------------------
    rng = _lcg(321)
    true = [i * 0.5 for i in range(50)]  # increasing ramp
    noisy = [t + (rng() / (1 << 24) - 0.5) * 3 for t in true]
    fit = isotonic_regression(noisy)
    err_raw = sum((a - b) ** 2 for a, b in zip(noisy, true))
    err_fit = sum((a - b) ** 2 for a, b in zip(fit, true))
    check("isotonic fit closer to true ramp than noisy data", err_fit < err_raw,
          f"fit {err_fit:.1f} vs raw {err_raw:.1f}")

    # ---- 9. hand example ----------------------------------------------------------------
    # classic: [1, 2, 0, 3] -> the 2,0 violate; pooled to 1 -> [1, 1, 1, 3]
    check("classic [1,2,0,3] -> [1,1,1,3]", isotonic_regression([1, 2, 0, 3]) == [1, 1, 1, 3])

    # ---- 10. calibration model interpolates and is monotone -----------------------------
    # uncalibrated scores vs binary-ish outcomes
    x = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
    y = [0, 0, 1, 0, 1, 1, 1]
    model = IsotonicModel(x, y)
    preds = model.predict(x)
    check("calibration predictions monotone", all(preds[i] <= preds[i + 1] + 1e-9 for i in range(len(preds) - 1)))
    check("calibration predict clamps below range", model.predict(-5) == model.predict(x[0]))
    check("calibration predict clamps above range", model.predict(5) == model.predict(x[-1]))
    mid = model.predict(0.55)
    check("interpolation between knots in range", preds[3] - 1e-9 <= mid <= preds[4] + 1e-9)

    # ---- 11. edge cases -----------------------------------------------------------------
    check("empty input -> empty", isotonic_regression([]) == [])
    check("single element unchanged", isotonic_regression([7.0]) == [7.0])
    try:
        isotonic_regression([1, 2], weights=[1])
        check("mismatched weights raise", False)
    except ValueError:
        check("mismatched weights raise", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

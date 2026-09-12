"""Tests for savitzky_golay: polynomial preservation, derivatives, noise reduction, gains."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savitzky_golay import coefficients, filter_signal

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 1234
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- coefficient gains -----------------------------------------------------
check("smoothing coefficients sum to 1 (unit DC gain)", abs(sum(coefficients(7, 3)) - 1.0) < 1e-12)
check("1st-derivative coefficients sum to 0", abs(sum(coefficients(7, 3, deriv=1))) < 1e-12)
check("2nd-derivative coefficients sum to 0", abs(sum(coefficients(7, 3, deriv=2))) < 1e-12)
# smoothing coefficients are symmetric
c = coefficients(9, 4)
check("smoothing coefficients are symmetric", all(abs(c[i] - c[-1 - i]) < 1e-12 for i in range(9)))
# 1st-derivative coefficients are antisymmetric
c1 = coefficients(9, 4, deriv=1)
check("1st-derivative coefficients are antisymmetric", all(abs(c1[i] + c1[-1 - i]) < 1e-12 for i in range(9)))

# --- a polynomial of degree <= filter degree passes unchanged --------------
def poly(x):
    return 3 + 2 * x - 0.5 * x ** 2 + 0.1 * x ** 3
y = [poly(x) for x in range(30)]
f = filter_signal(y, 7, 3)
check("degree-3 polynomial preserved by a degree-3 filter",
      max(abs(f[i] - y[i]) for i in range(30)) < 1e-9)

# a lower-degree polynomial too
lin = [2 * x + 1 for x in range(30)]
fl = filter_signal(lin, 5, 2)
check("linear signal preserved", max(abs(fl[i] - lin[i]) for i in range(30)) < 1e-9)

# --- derivative mode recovers the analytic derivative ----------------------
# d/dx(3 + 2x - 0.5x^2 + 0.1x^3) = 2 - x + 0.3x^2
d1 = filter_signal(y, 7, 3, deriv=1)
def dpoly(x):
    return 2 - x + 0.3 * x ** 2
check("1st derivative matches analytic (interior)",
      max(abs(d1[i] - dpoly(i)) for i in range(3, 27)) < 1e-8)

# second derivative: -1 + 0.6x
d2 = filter_signal(y, 7, 3, deriv=2)
def d2poly(x):
    return -1 + 0.6 * x
check("2nd derivative matches analytic (interior)",
      max(abs(d2[i] - d2poly(i)) for i in range(3, 27)) < 1e-7)

# --- derivative respects the sample spacing (delta) ------------------------
delta = 0.5
xs = [i * delta for i in range(30)]
ys = [poly(x) for x in xs]
d1s = filter_signal(ys, 7, 3, deriv=1, delta=delta)
check("derivative honors the delta spacing",
      max(abs(d1s[i] - dpoly(xs[i])) for i in range(3, 27)) < 1e-7)

# --- smoothing reduces noise -----------------------------------------------
clean = [math.sin(i * 0.15) for i in range(200)]
noisy = [clean[i] + (rng() - 0.5) * 0.4 for i in range(200)]
smoothed = filter_signal(noisy, 15, 3)
mse_noisy = sum((noisy[i] - clean[i]) ** 2 for i in range(200)) / 200
mse_smooth = sum((smoothed[i] - clean[i]) ** 2 for i in range(200)) / 200
check(f"smoothing reduces MSE to the clean signal ({mse_noisy:.4f} -> {mse_smooth:.4f})",
      mse_smooth < mse_noisy * 0.4)

# --- beats a moving average at preserving a Gaussian peak's height ---------
peak = [math.exp(-((i - 50) ** 2) / (2 * 5 ** 2)) for i in range(100)]
noisy_peak = [peak[i] + (rng() - 0.5) * 0.1 for i in range(100)]
sg = filter_signal(noisy_peak, 11, 3)
# moving average of the same window
half = 5
ma = []
for i in range(100):
    lo, hi = max(0, i - half), min(100, i + half + 1)
    ma.append(sum(noisy_peak[lo:hi]) / (hi - lo))
sg_peak = max(sg)
ma_peak = max(ma)
check(f"Savitzky-Golay preserves the peak height better than a moving average "
      f"(SG {sg_peak:.3f} vs MA {ma_peak:.3f}, true 1.0)", abs(sg_peak - 1.0) < abs(ma_peak - 1.0))

# --- output length matches input ------------------------------------------
check("output length equals input length", len(filter_signal(clean, 9, 2)) == len(clean))

# --- edge handling: constant signal stays constant -------------------------
const = [5.0] * 50
fc = filter_signal(const, 7, 2)
check("constant signal stays constant (including edges)", all(abs(v - 5.0) < 1e-9 for v in fc))

# --- validation ------------------------------------------------------------
raised = 0
for bad in [lambda: coefficients(4, 2), lambda: coefficients(5, 5), lambda: coefficients(5, 2, deriv=3)]:
    try:
        bad()
    except ValueError:
        raised += 1
check("invalid parameters raise ValueError", raised == 3)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all savitzky_golay tests passed")

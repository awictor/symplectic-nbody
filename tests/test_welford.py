"""Tests for welford.py -- online mean/variance and higher moments.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Results are cross-checked against
a two-pass computation, and the naive formula's instability is demonstrated.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import welford as W  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- matches two-pass on a known dataset -----------------------------------
data = [2, 4, 4, 4, 5, 5, 7, 9]     # mean 5, population variance 4
w = W.Welford().update(data)
m, v = W.two_pass_mean_variance(data)
check("count is correct", w.count() == 8)
check("mean matches two-pass", approx(w.mean, m, 1e-12))
check("population variance matches two-pass", approx(w.variance(), v, 1e-12))
check("mean is 5", approx(w.mean, 5.0, 1e-12))
check("population variance is 4", approx(w.variance(), 4.0, 1e-12))
check("std is 2", approx(w.std(), 2.0, 1e-12))
check("sample variance uses Bessel's correction (M2/(n-1))", approx(w.variance(1), 32.0 / 7.0, 1e-12))

# --- edge cases -------------------------------------------------------------
check("empty accumulator has zero variance", W.Welford().variance() == 0.0)
check("single value has zero variance", W.Welford().update([42]).variance() == 0.0)
check("single value mean is itself", W.Welford().update([42]).mean == 42.0)
check("constant data has zero variance", W.Welford().update([7] * 100).variance() == 0.0)

# --- matches two-pass over random data -------------------------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 8


gen = lcg(1)
ok = True
for _ in range(200):
    n = 2 + next(gen) % 300
    d = [(next(gen) / (1 << 24)) * 100 - 50 for _ in range(n)]
    wa = W.Welford().update(d)
    tm, tv = W.two_pass_mean_variance(d)
    if not (approx(wa.mean, tm, 1e-9 * max(1, abs(tm))) and approx(wa.variance(), tv, 1e-6 * max(1, tv))):
        ok = False
        break
check("mean and variance match two-pass over 200 random datasets", ok)

# --- numerical stability on offset data (the whole point) ------------------
offset = [1e9 + i for i in range(1, 6)]     # values 1e9+1 .. 1e9+5, variance = 2.0
check("Welford handles a huge offset correctly (var = 2)", approx(W.Welford().update(offset).variance(), 2.0, 1e-6))
check("the naive formula loses all precision here", W.naive_variance(offset) < 1.0)  # catastrophic cancellation
check("Welford variance is never negative on offset data", W.Welford().update(offset).variance() >= 0.0)
# an even harsher offset
harsh = [1e12 + x for x in (0.0, 1.0, 2.0, 3.0, 4.0)]
check("Welford stays exact at 1e12 offset", approx(W.Welford().update(harsh).variance(), 2.0, 1e-3))

# --- higher moments ---------------------------------------------------------
# symmetric data -> skewness ~ 0
sym = W.Welford().update([-3, -2, -1, 0, 1, 2, 3])
check("symmetric data has ~0 skewness", approx(sym.skewness(), 0.0, 1e-9))
# a right-skewed set has positive skewness
right = W.Welford().update([1, 1, 1, 1, 2, 3, 10])
check("right-skewed data has positive skewness", right.skewness() > 0)
# skewness/kurtosis match a direct computation
d = [1, 2, 2, 3, 3, 3, 4, 4, 5, 8]
wm = W.Welford().update(d)
mm = sum(d) / len(d)
m2 = sum((x - mm) ** 2 for x in d) / len(d)
m3 = sum((x - mm) ** 3 for x in d) / len(d)
m4 = sum((x - mm) ** 4 for x in d) / len(d)
check("skewness matches direct central moments", approx(wm.skewness(), m3 / m2 ** 1.5, 1e-9))
check("kurtosis matches direct central moments", approx(wm.kurtosis(), m4 / (m2 * m2), 1e-9))

# --- merge equals the concatenated stream ----------------------------------
gen2 = lcg(7)
d1 = [(next(gen2) / (1 << 24)) * 10 for _ in range(500)]
d2 = [(next(gen2) / (1 << 24)) * 20 + 5 for _ in range(300)]
wa, wb = W.Welford().update(d1), W.Welford().update(d2)
wmerge = wa.merge(wb)
wall = W.Welford().update(d1 + d2)
check("merged count equals total", wmerge.count() == 800)
check("merged mean equals the combined mean", approx(wmerge.mean, wall.mean, 1e-9))
check("merged variance equals the combined variance", approx(wmerge.variance(), wall.variance(), 1e-7))
check("merged skewness equals the combined skewness", approx(wmerge.skewness(), wall.skewness(), 1e-6))
check("merged kurtosis equals the combined kurtosis", approx(wmerge.kurtosis(), wall.kurtosis(), 1e-5))
# merging with an empty accumulator is a no-op
check("merge with empty is identity", approx(wa.merge(W.Welford()).variance(), wa.variance(), 1e-9))
# merge is order-independent for mean/variance
check("merge is symmetric in mean", approx(wa.merge(wb).mean, wb.merge(wa).mean, 1e-9))
check("merge is symmetric in variance", approx(wa.merge(wb).variance(), wb.merge(wa).variance(), 1e-7))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall welford tests passed")

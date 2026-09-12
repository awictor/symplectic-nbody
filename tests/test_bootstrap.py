"""Tests for bootstrap: resampling CIs, standard error, jackknife vs known quantities + coverage."""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bootstrap import (bootstrap_replicates, percentile_ci, bca_ci, bootstrap_se, jackknife,
                       mean, median, std, _percentile, _normal_ppf, _normal_cdf)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def uniform(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self, mu=0.0, sigma=1.0):
        return mu + sigma * (sum(self.uniform() for _ in range(12)) - 6.0)


def close(a, b, tol):
    return abs(a - b) <= tol


# --- normal-ppf / cdf round-trip -------------------------------------------
check("normal ppf/cdf round-trip at 0.5", close(_normal_ppf(0.5), 0.0, 1e-6))
check("normal ppf ~1.96 at 0.975", close(_normal_ppf(0.975), 1.959964, 1e-3))
check("normal cdf of ppf is identity", close(_normal_cdf(_normal_ppf(0.3)), 0.3, 1e-4))

# --- percentile helper ------------------------------------------------------
check("percentile of a range: median", close(_percentile(list(range(101)), 0.5), 50, 1e-9))
check("percentile 0 and 1 are the extremes",
      _percentile([1, 2, 3, 4], 0) == 1 and _percentile([1, 2, 3, 4], 1) == 4)

# --- bootstrap SE of the mean matches analytic s/sqrt(n) -------------------
rng = LCG(2026)
se_ok = True
for _ in range(30):
    data = [rng.normal(10, 2) for _ in range(150)]
    bse = bootstrap_se(data, mean, n_resamples=1500, seed=rng.s)
    analytic = std(data) / math.sqrt(len(data))
    if not close(bse, analytic, 0.05 * analytic + 0.01):
        se_ok = False
        break
check("bootstrap SE of the mean matches analytic s/sqrt(n) (30 datasets)", se_ok)

# --- jackknife is exact for the mean ---------------------------------------
data = [3.0, 5.0, 7.0, 11.0, 13.0]
_, bias, se = jackknife(data, mean)
check("jackknife bias of the mean is ~0", close(bias, 0.0, 1e-9))
check("jackknife SE of the mean equals s/sqrt(n)", close(se, std(data) / math.sqrt(len(data)), 1e-9))

# --- CIs bracket the estimate and lie within the data range ----------------
rng = LCG(4242)
bracket_ok = True
for _ in range(50):
    data = [rng.normal(5, 3) for _ in range(80)]
    theta = mean(data)
    for ci_fn in (percentile_ci, bca_ci):
        lo, hi = ci_fn(data, mean, n_resamples=1000, seed=rng.s)
        if not (lo <= theta <= hi):
            bracket_ok = False
            break
        if not (min(data) <= lo and hi <= max(data)):
            bracket_ok = False
            break
    if not bracket_ok:
        break
check("both CIs bracket the point estimate and lie within the data range", bracket_ok)

# --- CI ordering: lo <= hi, and higher confidence gives a wider interval ---
data = [rng.normal(0, 1) for _ in range(100)]
lo90, hi90 = percentile_ci(data, mean, confidence=0.90, seed=7)
lo99, hi99 = percentile_ci(data, mean, confidence=0.99, seed=7)
check("higher confidence gives a wider interval", (hi99 - lo99) >= (hi90 - lo90))
check("interval is ordered lo <= hi", lo90 <= hi90)

# --- reproducibility --------------------------------------------------------
d = [rng.normal(0, 1) for _ in range(50)]
check("same seed gives identical replicates",
      bootstrap_replicates(d, mean, 500, seed=99) == bootstrap_replicates(d, mean, 500, seed=99))

# --- coverage: a 90% interval covers the true mean about 90% of the time ---
rng = LCG(31337)
covered = 0
trials = 200
for t in range(trials):
    data = [rng.normal(10, 2) for _ in range(60)]
    lo, hi = percentile_ci(data, mean, confidence=0.90, n_resamples=600, seed=t * 7919 + 1)
    if lo <= 10 <= hi:
        covered += 1
rate = covered / trials
check(f"90% percentile interval covers the true mean ~90% ({covered}/{trials} = {rate:.2f})",
      0.80 <= rate <= 0.98)

# --- BCa coverage for a skewed statistic (variance of exponential-ish data)-
# BCa should cover better than percentile for skewed sampling distributions; just check it brackets
rng = LCG(555)
bca_ok = True
for _ in range(30):
    data = [-math.log(rng.uniform() + 1e-9) for _ in range(100)]     # exponential-ish
    lo, hi = bca_ci(data, mean, n_resamples=1000, seed=rng.s)
    if not (lo <= mean(data) <= hi):
        bca_ok = False
        break
check("BCa interval brackets the estimate on skewed data", bca_ok)

# --- works for the median (no closed-form SE) ------------------------------
data = [rng.normal(7, 1.5) for _ in range(120)]
lo, hi = bca_ci(data, median, n_resamples=1500, seed=1)
check("median CI is well-formed and brackets the sample median",
      lo <= median(data) <= hi and lo < hi)

# --- constant data: zero-width interval at the constant --------------------
lo, hi = percentile_ci([5.0] * 20, mean)
check("constant data gives a degenerate interval at the constant",
      close(lo, 5.0, 1e-9) and close(hi, 5.0, 1e-9))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bootstrap tests passed")

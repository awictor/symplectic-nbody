"""Tests for box_muller.py -- Gaussian sampling.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The output is verified against
the standard-normal moments and the 68-95-99.7 rule over large samples.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import box_muller as BM  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- the transform on known inputs -----------------------------------------
z0, z1 = BM.box_muller_pair(math.exp(-0.5), 0.0)   # r = 1, theta = 0 -> (1, 0)
check("box_muller_pair maps a known input exactly", approx(z0, 1.0, 1e-9) and approx(z1, 0.0, 1e-9))
z0, z1 = BM.box_muller_pair(math.exp(-0.5), 0.25)  # theta = pi/2 -> (0, 1)
check("box_muller_pair at quarter turn gives (0, 1)", approx(z0, 0.0, 1e-9) and approx(z1, 1.0, 1e-9))

# --- standard normal moments ------------------------------------------------
z = BM.standard_normals(200000, seed=1)
check("generates the requested count", len(z) == 200000)
check("mean is ~0", approx(BM.mean(z), 0.0, 0.02))
check("variance is ~1", approx(BM.variance(z), 1.0, 0.03))
check("skewness is ~0 (symmetric)", approx(BM.skewness(z), 0.0, 0.05))
check("kurtosis is ~3 (mesokurtic)", approx(BM.kurtosis(z), 3.0, 0.1))

# --- the 68-95-99.7 rule ----------------------------------------------------
check("~68% within 1 sd", approx(BM.fraction_within(z, 1), 0.6827, 0.01))
check("~95% within 2 sd", approx(BM.fraction_within(z, 2), 0.9545, 0.01))
check("~99.7% within 3 sd", approx(BM.fraction_within(z, 3), 0.9973, 0.005))

# --- odd count returns exactly n -------------------------------------------
check("odd count returns exactly n values", len(BM.standard_normals(101, seed=2)) == 101)
check("n=1 returns one value", len(BM.standard_normals(1, seed=2)) == 1)
check("n=0 returns empty", BM.standard_normals(0) == [])

# --- Marsaglia polar method has the same distribution ----------------------
p = BM.polar_normals(200000, seed=1)
check("polar mean is ~0", approx(BM.mean(p), 0.0, 0.02))
check("polar variance is ~1", approx(BM.variance(p), 1.0, 0.03))
check("polar kurtosis is ~3", approx(BM.kurtosis(p), 3.0, 0.1))
check("polar 68% within 1 sd", approx(BM.fraction_within(p, 1), 0.6827, 0.01))
check("polar returns exactly n values", len(BM.polar_normals(101, seed=3)) == 101)

# --- scaling to N(mu, sigma^2) ---------------------------------------------
s = BM.sample(200000, mu=10.0, sigma=2.0, seed=3)
check("scaled mean is ~mu", approx(BM.mean(s), 10.0, 0.05))
check("scaled variance is ~sigma^2", approx(BM.variance(s), 4.0, 0.1))
check("sigma=0 gives a constant sample", all(x == 5.0 for x in BM.sample(100, mu=5.0, sigma=0.0)))
check("scaled sample keeps standard-normal shape", approx(BM.skewness(s), 0.0, 0.05))
try:
    BM.sample(10, sigma=-1.0)
    check("rejects negative sigma", False)
except ValueError:
    check("rejects negative sigma", True)

# --- determinism ------------------------------------------------------------
check("same seed gives the same sample", BM.sample(100, seed=7) == BM.sample(100, seed=7))
check("different seeds differ", BM.sample(100, seed=1) != BM.sample(100, seed=2))

# --- descriptive helpers ----------------------------------------------------
check("mean of a known list", approx(BM.mean([1, 2, 3, 4]), 2.5, 1e-12))
check("variance of a constant list is 0", BM.variance([5, 5, 5]) == 0.0)
check("skewness of a symmetric list is ~0", approx(BM.skewness([-2, -1, 0, 1, 2]), 0.0, 1e-9))

# --- two independent halves are uncorrelated (rough) -----------------------
# split the sample and check the two halves' means are both ~0 (independence sanity)
half = len(z) // 2
check("both halves of the sample look standard-normal",
      approx(BM.mean(z[:half]), 0.0, 0.03) and approx(BM.mean(z[half:]), 0.0, 0.03))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall box_muller tests passed")

"""Tests for gibbs: bivariate/multivariate Gaussian moment recovery, discrete joint."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gibbs import (bivariate_gaussian, multivariate_gaussian, gibbs, sample_mean, sample_cov,
                   _invert)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- matrix inversion sanity -----------------------------------------------
M = [[4.0, 1.0], [1.0, 3.0]]
Minv = _invert(M)
prod = [[sum(M[i][k] * Minv[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
check("matrix inverse is correct (M * M^-1 = I)",
      all(abs(prod[i][j] - (1 if i == j else 0)) < 1e-9 for i in range(2) for j in range(2)))

# --- bivariate Gaussian: recover mean and covariance -----------------------
mean = [1.0, -2.0]
cov = [[2.0, 1.2], [1.2, 3.0]]
samples = bivariate_gaussian(mean, cov, 40000, burn_in=1000, seed=3)
m = sample_mean(samples)
c = sample_cov(samples)
check(f"bivariate sampled mean matches ({m[0]:.2f}, {m[1]:.2f})",
      abs(m[0] - 1.0) < 0.1 and abs(m[1] + 2.0) < 0.1)
check(f"bivariate variance of x matches ({c[0][0]:.2f} vs 2.0)", abs(c[0][0] - 2.0) < 0.2)
check(f"bivariate variance of y matches ({c[1][1]:.2f} vs 3.0)", abs(c[1][1] - 3.0) < 0.3)
check(f"bivariate covariance matches ({c[0][1]:.2f} vs 1.2)", abs(c[0][1] - 1.2) < 0.2)

# --- correlation recovered -------------------------------------------------
corr = c[0][1] / math.sqrt(c[0][0] * c[1][1])
true_corr = 1.2 / math.sqrt(2.0 * 3.0)
check(f"correlation matches ({corr:.3f} vs {true_corr:.3f})", abs(corr - true_corr) < 0.05)

# --- an uncorrelated Gaussian: near-zero covariance ------------------------
s_indep = bivariate_gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 40000, seed=7)
c_indep = sample_cov(s_indep)
check("independent Gaussian has near-zero covariance", abs(c_indep[0][1]) < 0.05)

# --- negative correlation --------------------------------------------------
s_neg = bivariate_gaussian([0.0, 0.0], [[1.0, -0.8], [-0.8, 1.0]], 40000, seed=9)
c_neg = sample_cov(s_neg)
check(f"negative correlation recovered ({c_neg[0][1]:.2f} vs -0.8)", abs(c_neg[0][1] + 0.8) < 0.08)

# --- 3-D multivariate Gaussian: full covariance ----------------------------
mean3 = [0.5, -1.0, 2.0]
cov3 = [[2.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.5]]
s3 = multivariate_gaussian(mean3, cov3, 40000, burn_in=2000, seed=5)
m3 = sample_mean(s3)
c3 = sample_cov(s3)
check("3-D mean recovered", all(abs(m3[i] - mean3[i]) < 0.1 for i in range(3)))
check("3-D diagonal (variances) recovered",
      all(abs(c3[i][i] - cov3[i][i]) < 0.2 for i in range(3)))
check("3-D off-diagonal covariances recovered",
      abs(c3[0][1] - 0.5) < 0.15 and abs(c3[0][2] - 0.3) < 0.15 and abs(c3[1][2] - 0.2) < 0.15)

# --- reproducibility -------------------------------------------------------
a = bivariate_gaussian([0, 0], [[1, 0.5], [0.5, 1]], 100, seed=11)
b = bivariate_gaussian([0, 0], [[1, 0.5], [0.5, 1]], 100, seed=11)
check("same seed gives identical samples", a == b)

# --- generic Gibbs on a discrete joint -------------------------------------
# a 2-variable joint over {0,1} x {0,1} with a known table; conditionals drawn from it
# P(x,y): favor (0,0) and (1,1) (positive correlation)
joint = {(0, 0): 0.4, (0, 1): 0.1, (1, 0): 0.1, (1, 1): 0.4}


def cond_x(state, rng):
    y = state[1]
    p0 = joint[(0, y)]
    p1 = joint[(1, y)]
    return 0 if rng.uniform() < p0 / (p0 + p1) else 1


def cond_y(state, rng):
    x = state[0]
    p0 = joint[(x, 0)]
    p1 = joint[(x, 1)]
    return 0 if rng.uniform() < p0 / (p0 + p1) else 1


ds = gibbs([0, 0], [cond_x, cond_y], 40000, burn_in=1000, seed=13)
from collections import Counter
counts = Counter((s[0], s[1]) for s in ds)
total = len(ds)
emp = {k: counts[k] / total for k in joint}
check("discrete Gibbs reproduces the joint (0,0)",
      abs(emp[(0, 0)] - 0.4) < 0.03)
check("discrete Gibbs reproduces the joint (1,1)",
      abs(emp[(1, 1)] - 0.4) < 0.03)
check("discrete Gibbs reproduces the off-diagonal cells",
      abs(emp[(0, 1)] - 0.1) < 0.03 and abs(emp[(1, 0)] - 0.1) < 0.03)

# --- samples are the requested count ---------------------------------------
check("returns exactly n_samples", len(bivariate_gaussian([0, 0], [[1, 0], [0, 1]], 500, seed=1)) == 500)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all gibbs tests passed")

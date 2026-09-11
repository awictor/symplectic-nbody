"""Tests for rejection_sampling.py -- von Neumann rejection sampling.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Sampled distributions are
verified against target moments, the acceptance-rate theory, and a histogram chi-square.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rejection_sampling as RS  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- a truncated Gaussian bump on [-4, 4] ----------------------------------
gauss = lambda x: math.exp(-0.5 * x * x)
s, acc = RS.sample_interval(gauss, -4, 4, 100000, seed=1)
check("returns the requested count", len(s) == 100000)
check("samples lie in the interval", all(-4 <= x <= 4 for x in s))
check("Gaussian sample mean is ~0", approx(RS.mean(s), 0.0, 0.02))
check("Gaussian sample variance is ~1", approx(RS.variance(s), 1.0, 0.03))
check("acceptance rate matches theory", approx(acc, RS.theoretical_acceptance(gauss, -4, 4), 0.02))
check("histogram matches the target (chi-square < 21.7 at 10 bins)",
      RS.chi_square_fit(s, gauss, -4, 4, 10) < 21.7)

# --- a triangular density f(x) = 2x on [0, 1] (mean 2/3) -------------------
tri = lambda x: 2 * x
st, acct = RS.sample_interval(tri, 0, 1, 100000, seed=2)
check("triangular sample mean is ~2/3", approx(RS.mean(st), 2 / 3, 0.01))
check("triangular acceptance is ~1/2 (area 1 under a max-2 box)", approx(acct, 0.5, 0.02))
check("triangular histogram matches (chi-square < 21.7 at 10 bins)",
      RS.chi_square_fit(st, tri, 0, 1, 10) < 21.7)

# --- unnormalized targets still sample correctly ---------------------------
un = lambda x: 7.3 * math.exp(-0.5 * x * x)      # same shape, arbitrary scale
su, _ = RS.sample_interval(un, -4, 4, 80000, seed=3)
check("unnormalized target: mean ~0", approx(RS.mean(su), 0.0, 0.02))
check("unnormalized target: variance ~1", approx(RS.variance(su), 1.0, 0.03))
check("scale factor does not change the sampled shape",
      RS.chi_square_fit(su, un, -4, 4, 10) < 21.7)

# --- a uniform target has ~100% acceptance ---------------------------------
uni = lambda x: 1.0
su2, accu = RS.sample_interval(uni, 0, 1, 20000, seed=4)
check("uniform target accepts nearly everything", accu > 0.95)
check("uniform sample mean is ~1/2", approx(RS.mean(su2), 0.5, 0.02))

# --- a tight envelope accepts more than a loose one ------------------------
_, acc_tight = RS.sample_interval(gauss, -4, 4, 20000, seed=5)             # m = max ~ 1
_, acc_loose = RS.sample_interval(gauss, -4, 4, 20000, m=5.0, seed=5)      # deliberately loose
check("a looser bound wastes more darts (lower acceptance)", acc_loose < acc_tight)
check("theoretical acceptance falls with a looser bound",
      RS.theoretical_acceptance(gauss, -4, 4, m=5.0) < RS.theoretical_acceptance(gauss, -4, 4))

# --- general rejection sampling with an arbitrary proposal -----------------
# target: unnormalized beta(3,2) shape x^2 (1-x) on [0, 1]; proposal: uniform
beta = lambda x: x * x * (1 - x) if 0 <= x <= 1 else 0.0
g_sampler = lambda rng: rng.random()
g_pdf = lambda x: 1.0
m = RS._estimate_max(beta, 0, 1) * 1.01
sg, accg = RS.sample(beta, g_sampler, g_pdf, m, 60000, seed=6)
check("beta(3,2) sample mean is ~0.6", approx(RS.mean(sg), 0.6, 0.01))
check("beta(3,2) histogram matches the target", RS.chi_square_fit(sg, beta, 0, 1, 10) < 21.7)
check("general acceptance rate is in (0, 1)", 0 < accg < 1)
try:
    RS.sample(beta, g_sampler, g_pdf, 0.0, 10)
    check("rejects nonpositive m", False)
except ValueError:
    check("rejects nonpositive m", True)

# --- determinism ------------------------------------------------------------
check("same seed gives the same samples",
      RS.sample_interval(gauss, -4, 4, 500, seed=9)[0] == RS.sample_interval(gauss, -4, 4, 500, seed=9)[0])

# --- helper sanity ----------------------------------------------------------
check("integrate of a unit box is 1", approx(RS._integrate(lambda x: 1.0, 0, 1), 1.0, 1e-9))
check("estimate_max finds the peak of a bump", approx(RS._estimate_max(gauss, -4, 4), 1.0, 1e-6))
check("histogram bins sum to the in-range count",
      sum(RS.histogram([0.1, 0.2, 0.9, 1.5], 0, 1, 4)) == 3)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall rejection_sampling tests passed")

"""Tests for ks_test: KS statistics vs brute ECDF gaps + statistical power/size checks."""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ks_test import (ks_one_sample, ks_two_sample, ks_one_sample_test, ks_two_sample_test,
                     empirical_cdf, uniform_cdf, normal_cdf, exponential_cdf,
                     brute_ks_one_sample, brute_ks_two_sample)

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
        # sum of 12 uniforms - 6 approximates N(0,1)
        return mu + sigma * (sum(self.uniform() for _ in range(12)) - 6.0)

    def exponential(self, rate=1.0):
        u = self.uniform()
        return -math.log(1 - u) / rate


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# --- empirical CDF ----------------------------------------------------------
F = empirical_cdf([1, 2, 3, 4])
check("empirical CDF is 0 below the min", F(0) == 0.0)
check("empirical CDF is 1 at/above the max", F(4) == 1.0 and F(10) == 1.0)
check("empirical CDF rises 1/n per point", close(F(2), 0.5))

# --- one-sample statistic matches brute ------------------------------------
rng = LCG(2026)
one_ok = True
for _ in range(200):
    n = 5 + rng.s % 50
    sample = [rng.uniform() for _ in range(n)]
    for cdf in (uniform_cdf(), normal_cdf(0.5, 0.3)):
        d = ks_one_sample(sample, cdf)
        bd = brute_ks_one_sample(sample, cdf)
        if not close(d, bd, 1e-3):
            one_ok = False
            print(f"  one-sample mismatch: {d} vs {bd}")
            break
    if not one_ok:
        break
check("one-sample KS statistic matches the brute ECDF gap (200 samples)", one_ok)

# --- two-sample statistic matches brute ------------------------------------
rng = LCG(4242)
two_ok = True
for _ in range(200):
    a = [rng.uniform() for _ in range(20 + rng.s % 40)]
    b = [rng.normal(0.5, 0.3) for _ in range(20 + rng.s % 40)]
    d = ks_two_sample(a, b)
    bd = brute_ks_two_sample(a, b)
    if not close(d, bd, 1e-6):
        two_ok = False
        print(f"  two-sample mismatch: {d} vs {bd}")
        break
check("two-sample KS statistic matches the brute ECDF gap (200 pairs)", two_ok)

# --- statistic properties ---------------------------------------------------
check("KS statistic is 0 for identical samples", ks_two_sample([1, 2, 3], [1, 2, 3]) == 0.0)
check("two-sample statistic is symmetric",
      close(ks_two_sample([1, 2, 3, 4], [2, 3, 5]), ks_two_sample([2, 3, 5], [1, 2, 3, 4])))
check("D lies in [0, 1]", 0 <= ks_two_sample([1, 2], [3, 4]) <= 1)
# completely separated samples -> D = 1
check("fully separated samples give D = 1", ks_two_sample([0, 1, 2], [10, 11, 12]) == 1.0)

# --- one-sample p-value: correct fit accepts, wrong fit rejects ------------
rng = LCG(777)
correct_accept = 0
correct_total = 0
for _ in range(60):
    sample = [rng.uniform() for _ in range(200)]
    _, p = ks_one_sample_test(sample, uniform_cdf())
    correct_total += 1
    if p > 0.05:
        correct_accept += 1
# a correct fit should NOT reject most of the time (size ~ 5%)
check(f"uniform sample vs uniform CDF rarely rejects ({correct_accept}/{correct_total} kept at 5%)",
      correct_accept >= 0.9 * correct_total)

wrong_reject = 0
for _ in range(60):
    sample = [rng.normal(0.5, 0.15) for _ in range(200)]
    _, p = ks_one_sample_test(sample, uniform_cdf())
    if p < 0.05:
        wrong_reject += 1
check(f"normal sample vs uniform CDF rejects with high power ({wrong_reject}/60)",
      wrong_reject >= 50)

# --- two-sample p-value: same dist accepts, different rejects --------------
rng = LCG(555)
same_accept = 0
for _ in range(60):
    a = [rng.uniform() for _ in range(150)]
    b = [rng.uniform() for _ in range(150)]
    _, p = ks_two_sample_test(a, b)
    if p > 0.05:
        same_accept += 1
check(f"two uniform samples rarely reject ({same_accept}/60 kept at 5%)", same_accept >= 54)

diff_reject = 0
for _ in range(60):
    a = [rng.uniform() for _ in range(150)]
    b = [rng.uniform() + 0.5 for _ in range(150)]      # shifted
    _, p = ks_two_sample_test(a, b)
    if p < 0.05:
        diff_reject += 1
check(f"shifted samples reject with high power ({diff_reject}/60)", diff_reject >= 55)

# --- p-value monotonicity: bigger D gives smaller p ------------------------
_, p_small = ks_two_sample_test([0, 1, 2, 3], [0.1, 1.1, 2.1, 3.1])
_, p_big = ks_two_sample_test([0, 1, 2, 3], [5, 6, 7, 8])
check("a larger KS statistic yields a smaller p-value", p_big <= p_small)

# --- known reference CDFs are valid distributions --------------------------
check("normal CDF is 0.5 at the mean", close(normal_cdf(0, 1)(0), 0.5))
check("exponential CDF is 1 - e^-1 at x = 1/rate", close(exponential_cdf(1)(1), 1 - math.exp(-1)))
check("uniform CDF is linear", close(uniform_cdf(0, 4)(1), 0.25))

# --- exponential sample fits exponential CDF -------------------------------
rng = LCG(31337)
exp_fit = 0
for _ in range(40):
    sample = [rng.exponential(1.5) for _ in range(200)]
    _, p = ks_one_sample_test(sample, exponential_cdf(1.5))
    if p > 0.05:
        exp_fit += 1
check(f"exponential sample fits its own CDF ({exp_fit}/40 kept)", exp_fit >= 34)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all ks_test tests passed")

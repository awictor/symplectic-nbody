"""Tests for coupon_collector.py -- the coupon collector's problem.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Analytic formulas are checked
against exact small cases and a seeded Monte-Carlo simulation.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import coupon_collector as cc  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


def rel(a, b, frac):
    return abs(a - b) <= frac * abs(b)


# --- harmonic numbers -------------------------------------------------------
check("H_1 = 1", approx(cc.harmonic(1), 1.0, 1e-12))
check("H_2 = 1.5", approx(cc.harmonic(2), 1.5, 1e-12))
check("H_4 = 25/12", approx(cc.harmonic(4), 25.0 / 12.0, 1e-12))
check("H_0 = 0", cc.harmonic(0) == 0.0)

# --- expected time exact cases ---------------------------------------------
check("E[T] for n=1 is 1", approx(cc.expected_time(1), 1.0, 1e-12))
check("E[T] for n=2 is 3", approx(cc.expected_time(2), 3.0, 1e-12))  # 2*(1+1/2)
check("E[T] for n=6 (a die) ~ 14.7", approx(cc.expected_time(6), 14.7, 0.05))
check("E[T] = n * H_n identity", approx(cc.expected_time(10), 10 * cc.harmonic(10), 1e-9))

# asymptotic tracks exact for moderate n
check("asymptotic within 1% of exact at n=50",
      rel(cc.expected_time_approx(50), cc.expected_time(50), 0.01))
check("asymptotic within 0.1% of exact at n=200",
      rel(cc.expected_time_approx(200), cc.expected_time(200), 0.001))

# --- variance ---------------------------------------------------------------
check("Var[T] positive", cc.variance(10) > 0)
check("Var[T] below the pi^2 n^2/6 bound", cc.variance(20) < math.pi ** 2 * 20 ** 2 / 6)
check("std dev is sqrt of variance", approx(cc.std_dev(15), math.sqrt(cc.variance(15)), 1e-12))
# n=1 is deterministic: exactly one draw, zero variance
check("Var[T] for n=1 is 0", approx(cc.variance(1), 0.0, 1e-12))

# --- partial collection -----------------------------------------------------
check("collecting all n equals expected_time",
      approx(cc.expected_partial(10, 10), cc.expected_time(10), 1e-9))
check("collecting 1 coupon takes 1 draw", approx(cc.expected_partial(10, 1), 1.0, 1e-9))
check("collecting 0 coupons takes 0 draws", approx(cc.expected_partial(10, 0), 0.0, 1e-12))
check("partial is increasing in k",
      cc.expected_partial(20, 5) < cc.expected_partial(20, 15))

# --- probability of completion ---------------------------------------------
# with n=6, exactly n draws complete only if all distinct: 6!/6^6
check("P(complete by exactly n) = n!/n^n",
      approx(cc.prob_complete_by(6, 6), math.factorial(6) / 6 ** 6, 1e-9))
check("P(complete by t < n) = 0", cc.prob_complete_by(6, 5) == 0.0)
check("P(complete) -> 1 for large t", approx(cc.prob_complete_by(6, 200), 1.0, 1e-6))
check("P(complete) is a probability in [0,1]", 0.0 <= cc.prob_complete_by(10, 30) <= 1.0)
check("P(complete) increases with t",
      cc.prob_complete_by(10, 20) < cc.prob_complete_by(10, 40))
# mean of the distribution from P(T > t) should reproduce E[T]: E[T] = sum_{t>=0} P(T > t)
n = 8
mean_from_cdf = sum(1.0 - cc.prob_complete_by(n, t) for t in range(0, 400))
check("E[T] recovered from the CDF (n=8)", rel(mean_from_cdf, cc.expected_time(n), 0.01))

# --- tail bound -------------------------------------------------------------
check("tail bound e^{-c} at c=0 is 1", approx(cc.tail_bound(10, 0.0), 1.0, 1e-12))
check("tail bound decreases with c", cc.tail_bound(10, 2.0) < cc.tail_bound(10, 1.0))
# the bound must actually hold: P(T > n ln n + c n) <= e^{-c}
nb, c = 20, 1.5
t = int(nb * math.log(nb) + c * nb)
check("tail bound holds against the exact CDF",
      (1.0 - cc.prob_complete_by(nb, t)) <= cc.tail_bound(nb, c) + 1e-9)

# --- Monte-Carlo agreement --------------------------------------------------
for nn in (6, 20):
    mean, sd = cc.simulate(nn, trials=4000, seed=7)
    check(f"simulated mean matches E[T] for n={nn}", rel(mean, cc.expected_time(nn), 0.05))
    check(f"simulated std matches sigma for n={nn}", rel(sd, cc.std_dev(nn), 0.10))

# m copies: simulation must exceed the single-copy time and rise with m
m1 = cc.simulate(15, trials=3000, seed=2, copies=1)[0]
m2 = cc.simulate(15, trials=3000, seed=2, copies=2)[0]
check("collecting 2 copies takes longer than 1", m2 > m1)
check("m-copies estimate is in the right ballpark of simulation",
      rel(cc.expected_time_m_copies(15, 2), m2, 0.25))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall coupon_collector tests passed")

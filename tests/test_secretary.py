"""Tests for secretary.py -- the secretary problem and the 1/e rule.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Analytic formulas are checked
against exact small cases and a seeded Monte-Carlo simulation.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import secretary as s  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


def rel(a, b, frac):
    return abs(a - b) <= frac * abs(b)


# --- win probability exact cases -------------------------------------------
check("taking the first candidate wins 1/n", approx(s.win_probability(100, 1), 0.01, 1e-12))
# n=2: r=1 gives 1/2, r=2 gives 1/2 (take the second) -- both 0.5
check("n=2, cutoff 1 wins 1/2", approx(s.win_probability(2, 1), 0.5, 1e-12))
check("n=2, cutoff 2 wins 1/2", approx(s.win_probability(2, 2), 0.5, 1e-12))
# n=3: the classic answer is r=2 with probability 1/2
check("n=3 optimal probability is 1/2", approx(s.win_probability(3, 2), 0.5, 1e-12))
check("win probability is in [0,1]", 0.0 <= s.win_probability(50, 20) <= 1.0)
try:
    s.win_probability(10, 0)
    check("rejects r out of range", False)
except ValueError:
    check("rejects r out of range", True)

# --- optimal cutoff ---------------------------------------------------------
r3, p3 = s.optimal_cutoff(3)
check("n=3 optimal cutoff is r=2", r3 == 2)
check("n=3 optimal probability is 1/2", approx(p3, 0.5, 1e-12))
r10, p10 = s.optimal_cutoff(10)
check("n=10 optimal cutoff is r=4", r10 == 4)
check("n=10 optimal probability ~ 0.399", approx(p10, 0.3987, 1e-3))
# the optimum really is a maximum: neighbours are no better
for n in (10, 25, 50):
    r, p = s.optimal_cutoff(n)
    left = s.win_probability(n, r - 1) if r > 1 else -1
    right = s.win_probability(n, r + 1) if r < n else -1
    check(f"cutoff is a local max for n={n}", p >= left and p >= right)

# --- 1/e asymptotics --------------------------------------------------------
inv_e = 1.0 / math.e
check("asymptotic probability is 1/e", approx(s.asymptotic_probability(), inv_e, 1e-12))
check("asymptotic fraction is 1/e", approx(s.asymptotic_fraction(), inv_e, 1e-12))
check("optimal fraction approaches 1/e for large n",
      approx(s.optimal_fraction(1000), inv_e, 0.01))
check("optimal probability approaches 1/e for large n",
      approx(s.optimal_cutoff(1000)[1], inv_e, 0.01))
# monotone-ish approach: n=1000 is closer to 1/e than n=10
check("larger n is closer to the 1/e probability",
      abs(s.optimal_cutoff(1000)[1] - inv_e) < abs(s.optimal_cutoff(10)[1] - inv_e))

# --- expected candidates seen ----------------------------------------------
r, _ = s.optimal_cutoff(100)
seen = s.expected_candidates_seen(100, r)
check("expected candidates seen is between the cutoff and n", r <= seen <= 100)
check("taking the first means seeing exactly 1", approx(s.expected_candidates_seen(50, 1), 1.0, 1e-9))

# --- Monte-Carlo agreement --------------------------------------------------
for n in (10, 50, 100):
    r, p = s.optimal_cutoff(n)
    emp = s.simulate(n, r, trials=8000, seed=5)
    check(f"simulation matches theory at the optimum for n={n}", rel(emp, p, 0.05))
# a suboptimal cutoff should do worse than the optimum in simulation too
r_opt, _ = s.optimal_cutoff(50)
emp_opt = s.simulate(50, r_opt, trials=8000, seed=9)
emp_first = s.simulate(50, 1, trials=8000, seed=9)  # take-the-first strategy
check("the optimal cutoff beats take-the-first in simulation", emp_opt > emp_first)
check("take-the-first simulates near 1/n", rel(emp_first, 1.0 / 50, 0.6))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall secretary tests passed")

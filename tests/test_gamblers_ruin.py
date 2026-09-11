"""Tests for gamblers_ruin.py -- the gambler's ruin problem.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Exact formulas are checked
against known values and a seeded Monte-Carlo simulation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gamblers_ruin as g  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


def rel(a, b, frac):
    return abs(a - b) <= frac * abs(b)


# --- fair game --------------------------------------------------------------
check("fair ruin from i is 1 - i/N", approx(g.ruin_probability(25, 100, 0.5), 0.75, 1e-12))
check("fair ruin from half is 1/2", approx(g.ruin_probability(50, 100, 0.5), 0.5, 1e-12))
check("reach target is i/N in a fair game", approx(g.reach_target_probability(25, 100, 0.5), 0.25, 1e-12))
check("ruin + reach = 1", approx(g.ruin_probability(30, 100) + g.reach_target_probability(30, 100), 1.0, 1e-12))
check("fair expected duration is i(N-i)", approx(g.expected_duration(25, 100, 0.5), 1875.0, 1e-9))
check("fair duration is symmetric i <-> N-i",
      approx(g.expected_duration(30, 100), g.expected_duration(70, 100), 1e-9))
check("duration maximal at the midpoint",
      g.expected_duration(50, 100) > g.expected_duration(10, 100))

# --- boundary conditions ----------------------------------------------------
check("starting broke means certain ruin", g.ruin_probability(0, 100) == 1.0)
check("starting at target means no ruin", g.ruin_probability(100, 100) == 0.0)
check("duration at a wall is 0", g.expected_duration(0, 100) == 0.0 and g.expected_duration(100, 100) == 0.0)
try:
    g.ruin_probability(5, 100, 0.0)
    check("rejects p=0", False)
except ValueError:
    check("rejects p=0", True)
try:
    g.ruin_probability(150, 100)
    check("rejects i > N", False)
except ValueError:
    check("rejects i > N", True)

# --- biased game ------------------------------------------------------------
# a tiny edge against the player makes ruin much more likely than the fair 1/2
r49 = g.ruin_probability(50, 100, 0.49)
check("even-stakes but p=0.49 gives ruin ~0.88", approx(r49, 0.8808, 1e-3))
check("adverse edge raises ruin above the fair value", r49 > 0.5)
# a favorable game lowers ruin below fair
check("favorable p=0.6 lowers ruin below 1/2", g.ruin_probability(50, 100, 0.6) < 0.5)
# biased formula must still give ruin+reach = 1
check("biased ruin + reach = 1",
      approx(g.ruin_probability(40, 100, 0.55) + g.reach_target_probability(40, 100, 0.55), 1.0, 1e-12))
# small exact biased case p=0.6, i=5, N=10
check("p=0.6, i=5, N=10 ruin ~0.1164", approx(g.ruin_probability(5, 10, 0.6), 0.1164, 1e-3))
check("biased duration is positive", g.expected_duration(50, 100, 0.49) > 0)

# --- infinite house ---------------------------------------------------------
check("fair game vs infinite house is certain ruin", g.ruin_probability_infinite_house(10, 0.5) == 1.0)
check("unfavorable game vs infinite house is certain ruin", g.ruin_probability_infinite_house(10, 0.4) == 1.0)
check("favorable game vs infinite house is (q/p)^i",
      approx(g.ruin_probability_infinite_house(10, 0.6), (0.4 / 0.6) ** 10, 1e-9))
check("larger stake escapes the infinite house more often",
      g.ruin_probability_infinite_house(20, 0.6) < g.ruin_probability_infinite_house(5, 0.6))
# the finite-N ruin should approach the infinite-house value as N grows (favorable game)
check("finite ruin -> infinite-house limit as N grows",
      approx(g.ruin_probability(10, 5000, 0.6), g.ruin_probability_infinite_house(10, 0.6), 1e-3))

# --- Monte-Carlo agreement --------------------------------------------------
cases = [(25, 100, 0.5), (50, 100, 0.49), (5, 10, 0.6)]
for i, N, p in cases:
    rf, md = g.simulate(i, N, p, trials=8000, seed=3)
    check(f"simulated ruin matches theory (i={i},N={N},p={p})",
          approx(rf, g.ruin_probability(i, N, p), 0.03))
    check(f"simulated duration matches theory (i={i},N={N},p={p})",
          rel(md, g.expected_duration(i, N, p), 0.08))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall gamblers_ruin tests passed")

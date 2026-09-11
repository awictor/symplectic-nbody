"""Tests for kelly.py -- the Kelly criterion.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The growth rate is checked
against the analytic optimum and a seeded Monte-Carlo of the compounding bankroll.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import kelly as k  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- optimal fraction -------------------------------------------------------
check("even-money p=0.6 gives f* = 0.2", approx(k.optimal_fraction(0.6, 1.0), 0.2, 1e-12))
check("f* = p - q/b identity", approx(k.optimal_fraction(0.6, 1.0), 0.6 - 0.4 / 1.0, 1e-12))
check("b=2 odds, p=0.5 gives f* = 0.25", approx(k.optimal_fraction(0.5, 2.0), 0.25, 1e-12))
check("fair even-money bet stakes nothing", approx(k.optimal_fraction(0.5, 1.0), 0.0, 1e-12))
check("a losing bet stakes nothing (clamped to 0)", k.optimal_fraction(0.4, 1.0) == 0.0)
check("a near-certain bet stakes almost everything", k.optimal_fraction(0.99, 1.0) > 0.9)
check("f* never exceeds 1", k.optimal_fraction(0.999, 1.0) <= 1.0)
try:
    k.optimal_fraction(1.2)
    check("rejects p out of range", False)
except ValueError:
    check("rejects p out of range", True)
try:
    k.optimal_fraction(0.6, 0.0)
    check("rejects nonpositive odds", False)
except ValueError:
    check("rejects nonpositive odds", True)

# --- edge -------------------------------------------------------------------
check("edge of p=0.6 even money is 0.2", approx(k.edge(0.6, 1.0), 0.2, 1e-12))
check("fair bet has zero edge", approx(k.edge(0.5, 1.0), 0.0, 1e-12))
check("positive edge <=> positive Kelly fraction",
      (k.edge(0.55) > 0) == (k.optimal_fraction(0.55) > 0))

# --- growth rate: f* is the maximum ----------------------------------------
p, b = 0.6, 1.0
gstar = k.optimal_growth_rate(p, b)
fstar = k.optimal_fraction(p, b)
check("growth at f=0 is zero", approx(k.growth_rate(0.0, p, b), 0.0, 1e-12))
check("optimal growth rate is positive for an edge", gstar > 0)
check("no other fraction beats f* (below)", k.growth_rate(fstar - 0.05, p, b) < gstar)
check("no other fraction beats f* (above)", k.growth_rate(fstar + 0.05, p, b) < gstar)
# scan a grid: f* is the argmax
best = max((k.growth_rate(i / 1000, p, b), i / 1000) for i in range(1, 1000))
check("grid search confirms f* is the argmax", approx(best[1], fstar, 2e-3))

# --- overbetting and break-even --------------------------------------------
check("overbetting past break-even loses money", k.growth_rate(0.6, p, b) < 0)
be = k.break_even_fraction(p, b)
check("break-even fraction is beyond f*", be > fstar)
check("growth at the break-even fraction is ~0", approx(k.growth_rate(be, p, b), 0.0, 1e-6))
check("break-even is near 2 f* for a small edge",
      approx(k.break_even_fraction(0.52, 1.0), 2 * k.optimal_fraction(0.52, 1.0), 0.01))

# --- fractional Kelly -------------------------------------------------------
half = k.growth_rate(fstar / 2, p, b)
check("half-Kelly still grows", half > 0)
check("half-Kelly gives ~3/4 of the full growth", approx(half / gstar, 0.75, 0.05))
check("half-Kelly grows slower than full Kelly", half < gstar)

# --- doubling time ----------------------------------------------------------
check("doubling time is ln2 / g*", approx(k.doubling_time(p, b), math.log(2) / gstar, 1e-9))
check("a fair bet never doubles (infinite time)", k.doubling_time(0.5, 1.0) == float("inf"))
check("bigger edge doubles faster", k.doubling_time(0.7) < k.doubling_time(0.6))

# --- Monte-Carlo agreement --------------------------------------------------
check("simulated growth at f* matches theory",
      approx(k.simulate_growth(fstar, p, b, bets=500, trials=3000, seed=7), gstar, 3e-3))
check("simulated growth at half-Kelly matches theory",
      approx(k.simulate_growth(fstar / 2, p, b, bets=500, trials=3000, seed=7), half, 3e-3))
# the simulated bankroll grows fastest at f* among a few fractions
sims = {f: k.simulate_growth(f, p, b, bets=400, trials=2500, seed=11)
        for f in (0.1, 0.2, 0.35, 0.5)}
check("simulation grows fastest at the Kelly fraction", max(sims, key=sims.get) == 0.2)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall kelly tests passed")

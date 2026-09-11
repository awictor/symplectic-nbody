"""Tests for birthday.py -- the birthday problem.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Analytic formulas are checked
against known exact values and a seeded Monte-Carlo simulation.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import birthday as b  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, c, tol):
    return abs(a - c) <= tol


def rel(a, c, frac):
    return abs(a - c) <= frac * abs(c)


# --- distinct / collision probabilities ------------------------------------
check("one person never collides", b.prob_collision(1) == 0.0)
check("zero people never collide", b.prob_collision(0) == 0.0)
check("one person is trivially distinct", b.prob_distinct(1) == 1.0)
check("distinct + collision = 1", approx(b.prob_distinct(30) + b.prob_collision(30), 1.0, 1e-12))
# the famous numbers
check("23 people cross 50% (P ~ 0.507)", approx(b.prob_collision(23), 0.5073, 1e-3))
check("22 people are below 50%", b.prob_collision(22) < 0.5)
check("23 people are above 50%", b.prob_collision(23) > 0.5)
check("collision probability increases with k", b.prob_collision(30) > b.prob_collision(23))
check("probabilities are in [0,1]", 0.0 <= b.prob_collision(40) <= 1.0)
try:
    b.prob_distinct(-1)
    check("rejects negative k", False)
except ValueError:
    check("rejects negative k", True)

# pigeonhole: more people than days forces a collision
check("366 people must share a birthday (365 days)", b.prob_collision(366) == 1.0)
check("366 distinct is impossible", b.prob_distinct(366) == 0.0)
# small day-count exact case: d=2 (a coin), 2 people distinct with prob 1/2
check("d=2, two people distinct with prob 1/2", approx(b.prob_distinct(2, days=2), 0.5, 1e-12))
check("d=2, three people always collide", b.prob_collision(3, days=2) == 1.0)

# --- minimum group size -----------------------------------------------------
check("min group for 50% collision is 23", b.min_people_for(0.5) == 23)
check("min group for 99% collision is 57", b.min_people_for(0.99) == 57)
check("min group for 99.9% collision is 70", b.min_people_for(0.999) == 70)
check("min group for zero probability is 0", b.min_people_for(0.0) == 0)
check("min group actually reaches the target",
      b.prob_collision(b.min_people_for(0.9)) >= 0.9)
check("one fewer misses the target",
      b.prob_collision(b.min_people_for(0.9) - 1) < 0.9)

# --- median / expected first collision -------------------------------------
check("median approximation ~ 22.5 for 365 days", approx(b.median_collision(), 22.49, 0.1))
check("median ~ 1.177 sqrt(d)", rel(b.median_collision(), 1.1774 * math.sqrt(365), 0.001))
check("expected first collision ~ 24 for 365 days", approx(b.expected_first_collision(), 23.94, 0.1))
check("expected first collision ~ sqrt(pi d / 2)",
      approx(b.expected_first_collision(), math.sqrt(math.pi * 365 / 2), 1e-9))

# --- Poisson approximation --------------------------------------------------
check("approx close to exact at k=23", approx(b.collision_approx(23), b.prob_collision(23), 0.02))
check("approx close to exact at k=40", approx(b.collision_approx(40), b.prob_collision(40), 0.02))
check("approx is a probability", 0.0 <= b.collision_approx(50) <= 1.0)

# --- hash / birthday attack -------------------------------------------------
# a b-bit hash collides after ~2^(b/2) tries: 128-bit near 2^64
check("128-bit hash collides near 2^64 tries", rel(b.hash_collision_tries(128), 2.0 ** 64, 0.3))
check("collision cost is ~sqrt of the preimage cost",
      rel(b.hash_collision_tries(64) ** 2, math.pi / 2 * 2.0 ** 64, 1e-6))
check("more bits means more tries", b.hash_collision_tries(256) > b.hash_collision_tries(128))

# --- Monte-Carlo agreement --------------------------------------------------
for k in (10, 23, 40):
    emp = b.simulate_collision(k, trials=10000, seed=5)
    check(f"simulated collision prob matches theory at k={k}",
          approx(emp, b.prob_collision(k), 0.03))
emp_first = b.simulate_first_collision(trials=10000, seed=3)
check("simulated first-collision mean matches theory",
      rel(emp_first, b.expected_first_collision(), 0.08))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall birthday tests passed")

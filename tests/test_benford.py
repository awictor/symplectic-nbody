"""Tests for benford.py -- Benford's leading-digit law.

Self-running: prints PASS/FAIL per check, exits 1 if any fail.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import benford as b  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, c, tol):
    return abs(a - c) <= tol


# --- the law itself ---------------------------------------------------------
dist = b.benford_distribution()
check("nine first-digit probabilities", len(dist) == 9)
check("probabilities sum to 1", approx(sum(dist), 1.0, 1e-12))
check("P(1) ~ 0.301", approx(b.benford_probability(1), 0.30103, 1e-4))
check("P(9) ~ 0.0458", approx(b.benford_probability(9), 0.045757, 1e-4))
check("digit 1 is the most common", dist[0] == max(dist))
check("digit 9 is the least common", dist[8] == min(dist))
check("probabilities strictly decreasing 1..9",
      all(dist[i] > dist[i + 1] for i in range(8)))
try:
    b.benford_probability(0)
    check("rejects digit 0", False)
except ValueError:
    check("rejects digit 0", True)

# --- second-position digit law ---------------------------------------------
pos2 = [b.digit_probability(d, 2) for d in range(10)]
check("second-position probabilities sum to 1", approx(sum(pos2), 1.0, 1e-9))
check("second position closer to uniform than first",
      (max(pos2) - min(pos2)) < (max(dist) - min(dist)))
# high positions -> uniform 0.1
check("fifth-position digit ~ uniform 0.1", approx(b.digit_probability(3, 5), 0.1, 1e-3))

# --- leading digit extraction ----------------------------------------------
check("leading digit of 0.0037 is 3", b.leading_digit(0.0037) == 3)
check("leading digit of 920 is 9", b.leading_digit(920) == 9)
check("leading digit of 1.0 is 1", b.leading_digit(1.0) == 1)
check("leading digit of 9.99 is 9", b.leading_digit(9.99) == 9)
check("leading digit of a negative uses magnitude", b.leading_digit(-521) == 5)
check("leading digit of 0 is 0", b.leading_digit(0) == 0)
# huge integer (no float overflow): 300! begins with 3
fac = b.factorials(300)
check("leading digit of 300! (huge int) is 3", b.leading_digit(fac[-1]) == 3)

# --- histograms -------------------------------------------------------------
counts = b.leading_digit_counts([1, 1, 2, 30, 400])
check("counts tally leading digits", counts == [2, 1, 1, 1, 0, 0, 0, 0, 0])
check("exact zeros are skipped", b.leading_digit_counts([0, 0, 5]) == [0, 0, 0, 0, 1, 0, 0, 0, 0])
freqs = b.leading_digit_frequencies([1, 2])
check("frequencies sum to 1", approx(sum(freqs), 1.0, 1e-12))

# --- classic Benford sequences pass, uniform fails --------------------------
fib = b.fibonacci(500)
check("Fibonacci follows Benford", b.follows_benford(fib))
check("Fibonacci chi-square is small", b.chi_square(fib) < 5.0)
check("Fibonacci total-variation distance is small", b.total_variation_distance(fib) < 0.05)

p2 = b.powers(2, 500)
check("powers of 2 follow Benford", b.follows_benford(p2))
check("factorials follow Benford", b.follows_benford(fac))

# a dataset with every leading digit equally common (uniform) must be rejected
uniform = []
for decade in (1, 10, 100):
    for d in range(1, 10):
        uniform += [d * decade] * 50
check("a uniform-leading-digit dataset fails Benford", not b.follows_benford(uniform))
check("uniform dataset has a large chi-square", b.chi_square(uniform) > 15.51)
check("uniform dataset has a large TV distance", b.total_variation_distance(uniform) > 0.1)

# empty data is well-behaved
check("empty data gives zero chi-square", b.chi_square([]) == 0.0)
check("empty data gives zero counts", b.leading_digit_counts([]) == [0] * 9)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall benford tests passed")

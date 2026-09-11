"""Tests for alias_method.py -- Walker-Vose alias sampling.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The empirical distribution is
verified against the target weights with chi-square tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import alias_method as A  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- construction & normalization ------------------------------------------
s = A.AliasSampler([1, 1, 1, 1], seed=5)
check("weights normalize to probabilities", all(approx(p, 0.25, 1e-12) for p in s.probabilities()))
check("probabilities sum to 1", approx(sum(s.probabilities()), 1.0, 1e-12))
check("unnormalized weights are accepted", approx(sum(A.AliasSampler([2, 4, 4]).probabilities()), 1.0, 1e-12))
check("target matches the input ratio", approx(A.AliasSampler([10, 3, 1]).probabilities()[0], 10 / 14, 1e-12))
try:
    A.AliasSampler([])
    check("rejects empty weights", False)
except ValueError:
    check("rejects empty weights", True)
try:
    A.AliasSampler([1, -1])
    check("rejects negative weights", False)
except ValueError:
    check("rejects negative weights", True)
try:
    A.AliasSampler([0, 0])
    check("rejects all-zero weights", False)
except ValueError:
    check("rejects all-zero weights", True)

# --- alias table structure --------------------------------------------------
check("each column's prob is in [0, 1]", all(0 <= p <= 1 + 1e-9 for p in s.prob))
check("aliases are valid indices", all(0 <= a < s.n for a in s.alias))

# --- sampling produces valid indices ---------------------------------------
s3 = A.AliasSampler([5, 3, 2, 1, 4], seed=1)
check("samples are valid outcome indices", all(0 <= s3.sample() < 5 for _ in range(2000)))
check("sample_many returns the requested count", len(s3.sample_many(500)) == 500)
check("a single-outcome sampler always returns 0", all(A.AliasSampler([7]).sample() == 0 for _ in range(50)))

# --- determinism ------------------------------------------------------------
a1 = A.AliasSampler([3, 1, 2], seed=9).sample_many(100)
a2 = A.AliasSampler([3, 1, 2], seed=9).sample_many(100)
check("same seed gives the same draws", a1 == a2)
check("different seeds usually differ", A.AliasSampler([3, 1, 2], seed=1).sample_many(50) !=
      A.AliasSampler([3, 1, 2], seed=2).sample_many(50))

# --- the empirical distribution matches the weights (chi-square) -----------
uni = A.AliasSampler([1, 1, 1, 1, 1], seed=1)
check("uniform sampler passes chi-square (4 dof, 1% ~ 13.3)", A.chi_square(uni, 50000) < 13.3)
skew = A.AliasSampler([10, 3, 1], seed=2)
check("skewed sampler passes chi-square (2 dof, 1% ~ 9.2)", A.chi_square(skew, 60000) < 9.2)
emp = A.empirical_distribution(skew, 60000)
target = skew.probabilities()
check("empirical frequencies match the target weights", all(approx(emp[i], target[i], 0.02) for i in range(3)))

# --- larger distribution ----------------------------------------------------
def lcg(seed):
    st = seed
    while True:
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        yield st >> 16


gen = lcg(7)
big_w = [1 + next(gen) % 30 for _ in range(25)]
big = A.AliasSampler(big_w, seed=3)
check("25-outcome sampler passes chi-square (24 dof, 1% ~ 43)", A.chi_square(big, 120000) < 43)
check("empirical sums to 1", approx(sum(A.empirical_distribution(big, 20000)), 1.0, 1e-9))

# --- a highly skewed distribution (one dominant outcome) -------------------
dom = A.AliasSampler([1000, 1, 1, 1], seed=4)
freq = A.empirical_distribution(dom, 40000)
check("dominant outcome sampled ~99.7% of the time", approx(freq[0], 1000 / 1003, 0.01))
check("rare outcomes are still occasionally sampled", freq[1] > 0)

# --- an outcome with zero weight is never sampled --------------------------
withzero = A.AliasSampler([5, 0, 5], seed=1)
check("zero-weight outcome is never sampled", A.empirical_distribution(withzero, 20000)[1] == 0.0)
check("zero-weight target probability is 0", withzero.probabilities()[1] == 0.0)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall alias_method tests passed")

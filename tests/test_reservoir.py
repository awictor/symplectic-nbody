"""Tests for reservoir.py -- reservoir sampling.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Uniformity is verified
statistically with a chi-square test over many independent runs.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import reservoir as R  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- basic behaviour --------------------------------------------------------
s = R.sample(range(10), 3, seed=5)
check("sample returns k items", len(s) == 3)
check("sampled items come from the stream", all(0 <= x < 10 for x in s))
check("sample has no duplicates", len(set(s)) == 3)
check("fewer than k items returns all of them", sorted(R.sample([1, 2], 5)) == [1, 2])
check("k=0 returns an empty sample", R.sample(range(10), 0) == [])
check("sampling from an empty stream is empty", R.sample([], 3) == [])
check("k equal to n returns everything", sorted(R.sample(range(4), 4)) == [0, 1, 2, 3])
try:
    R.sample(range(5), -1)
    check("rejects negative k", False)
except ValueError:
    check("rejects negative k", True)

# --- determinism ------------------------------------------------------------
check("same seed gives the same sample", R.sample(range(100), 5, seed=7) == R.sample(range(100), 5, seed=7))
check("different seeds usually differ", R.sample(range(100), 5, seed=1) != R.sample(range(100), 5, seed=2))

# --- uniformity: every element is picked with probability k/n --------------
n, k, trials = 10, 3, 40000
counts = R.selection_frequencies(n, k, trials, seed=1)
expected = trials * k / n
check("total selections equal trials * k", sum(counts) == trials * k)
chi = R.chi_square_uniformity(counts, expected)
# 9 degrees of freedom, 1% critical value ~ 21.7; a uniform sampler is comfortably under
check("selection counts pass a chi-square uniformity test (chi2 < 21.7)", chi < 21.7)
check("every element gets selected roughly k/n of the time",
      all(abs(c - expected) < 0.1 * expected for c in counts))

# --- k=1 uniform pick -------------------------------------------------------
picks = Counter()
for t in range(20000):
    picks[R.sample(range(5), 1, seed=t)[0]] += 1
chi1 = R.chi_square_uniformity([picks[i] for i in range(5)], 20000 / 5)
check("k=1 uniform single pick passes chi-square", chi1 < 13.3)  # 4 dof, 1% ~ 13.3

# --- streaming Reservoir object --------------------------------------------
res = R.Reservoir(3, seed=7).update(range(100))
check("streaming reservoir holds k items", len(res.sample()) == 3)
check("streaming reservoir items are from the stream", all(0 <= x < 100 for x in res.sample()))
check("streaming reservoir tracks the count", res.n == 100)
check("streaming matches one-shot sample for the same seed",
      R.Reservoir(4, seed=3).update(range(50)).sample() == R.sample(range(50), 4, seed=3))
# a streaming reservoir with fewer than k items holds them all
small = R.Reservoir(10, seed=1).update(range(4))
check("streaming reservoir with < k items holds all", sorted(small.sample()) == [0, 1, 2, 3])

# --- streaming reservoir is uniform too ------------------------------------
scounts = [0] * 8
for t in range(20000):
    for x in R.Reservoir(2, seed=t).update(range(8)).sample():
        scounts[x] += 1
chi_s = R.chi_square_uniformity(scounts, 20000 * 2 / 8)
check("streaming reservoir passes chi-square uniformity", chi_s < 18.5)  # 7 dof, 1% ~ 18.5

# --- weighted reservoir sampling -------------------------------------------
check("weighted sample returns k items", len(R.weighted_sample(["a", "b", "c", "d"], [1, 1, 1, 1], 2, seed=1)) == 2)
check("weighted sample has no duplicates",
      len(set(R.weighted_sample(list(range(10)), [1] * 10, 4, seed=2))) == 4)
check("weighted k=0 is empty", R.weighted_sample(["a"], [1], 0) == [])
try:
    R.weighted_sample(["a", "b"], [1], 1)
    check("weighted rejects mismatched lengths", False)
except ValueError:
    check("weighted rejects mismatched lengths", True)
try:
    R.weighted_sample(["a"], [0], 1)
    check("weighted rejects nonpositive weights", False)
except ValueError:
    check("weighted rejects nonpositive weights", True)
# a heavily-weighted item is sampled far more often
picks = Counter()
for s in range(3000):
    for x in R.weighted_sample(["a", "b", "c"], [1, 1, 10], 1, seed=s):
        picks[x] += 1
check("heavily-weighted item dominates the sample", picks["c"] > picks["a"] + picks["b"])
check("weighted frequency roughly tracks the weights (c ~ 10/12)",
      abs(picks["c"] / 3000 - 10 / 12) < 0.08)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall reservoir tests passed")

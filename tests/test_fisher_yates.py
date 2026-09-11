"""Tests for fisher_yates.py -- unbiased shuffling.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Uniformity is verified by
enumerating every permutation over many trials; the naive shuffle's bias is demonstrated.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fisher_yates as FY  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- shuffle produces a valid permutation ----------------------------------
orig = list(range(10))
sh = FY.shuffle(orig, seed=5)
check("shuffle returns a permutation", FY.is_permutation(orig, sh))
check("shuffle does not modify the input", orig == list(range(10)))
check("shuffle of a single element is itself", FY.shuffle([42]) == [42])
check("shuffle of empty is empty", FY.shuffle([]) == [])
check("same seed gives the same shuffle", FY.shuffle(orig, seed=7) == FY.shuffle(orig, seed=7))
check("different seeds usually differ", FY.shuffle(orig, seed=1) != FY.shuffle(orig, seed=2))
check("shuffle works on strings/labels",
      FY.is_permutation(list("abcdef"), FY.shuffle(list("abcdef"), seed=3)))

# --- uniformity: every permutation is equally likely -----------------------
n, trials = 4, 60000
counts = FY.permutation_counts(n, trials, FY.shuffle, seed=1)
nfact = math.factorial(n)
check("every one of n! permutations appears", len(counts) == nfact)
chi = FY.chi_square_uniform(counts, nfact, trials)
# 23 degrees of freedom, 1% critical value ~ 41.6
check("Fisher-Yates passes a chi-square uniformity test (chi2 < 41.6)", chi < 41.6)
expected = trials / nfact
check("each permutation appears ~ trials/n! times",
      all(abs(c - expected) < 0.15 * expected for c in counts.values()))

# --- the naive shuffle is demonstrably biased ------------------------------
naive_counts = FY.permutation_counts(n, trials, FY.naive_shuffle, seed=1)
naive_chi = FY.chi_square_uniform(naive_counts, nfact, trials)
check("the naive shuffle is biased (huge chi-square)", naive_chi > 100)
check("Fisher-Yates is far more uniform than naive", chi < naive_chi / 10)
# naive is still a permutation, just not uniform
check("naive shuffle still returns a permutation", FY.is_permutation(orig, FY.naive_shuffle(orig, seed=1)))

# --- sample without replacement --------------------------------------------
smp = FY.sample_without_replacement(range(20), 5, seed=3)
check("sample has the requested size", len(smp) == 5)
check("sample has no duplicates", len(set(smp)) == 5)
check("sample elements come from the population", all(0 <= x < 20 for x in smp))
check("k=0 sample is empty", FY.sample_without_replacement(range(10), 0) == [])
check("k=n sample is a full permutation", FY.is_permutation(list(range(6)), FY.sample_without_replacement(range(6), 6, seed=1)))
try:
    FY.sample_without_replacement(range(5), 9)
    check("rejects k > n", False)
except ValueError:
    check("rejects k > n", True)
# the k-sample is uniform over which elements appear (each element ~ k/n of the time)
freq = [0] * 8
for t in range(24000):
    for x in FY.sample_without_replacement(range(8), 3, seed=t):
        freq[x] += 1
chi_s = FY.chi_square_uniform({(i,): freq[i] for i in range(8)}, 8, 24000 * 3)
check("sampled elements are chosen uniformly", chi_s < 20.1)  # 7 dof, 1% ~ 18.5-ish, allow slack

# --- Sattolo's cyclic shuffle ----------------------------------------------
for s in range(20):
    cyc = FY.sattolo_cycle(list(range(7)), seed=s)
    if not FY.is_permutation(list(range(7)), cyc):
        check("sattolo returns a permutation", False)
        break
else:
    check("sattolo returns a permutation", True)
check("sattolo produces a single n-cycle", all(
    FY.is_single_cycle(FY.sattolo_cycle(list(range(8)), seed=s)) for s in range(30)))
check("the identity is NOT a single cycle", not FY.is_single_cycle([0, 1, 2, 3]))
check("a proper cycle is detected", FY.is_single_cycle([1, 2, 3, 0]))
# sattolo never leaves an element fixed
no_fixed = all(
    all(FY.sattolo_cycle(list(range(6)), seed=s)[i] != i for i in range(6))
    for s in range(30)
)
check("sattolo never leaves a fixed point", no_fixed)

# --- is_permutation / is_single_cycle helpers ------------------------------
check("is_permutation catches a missing element", not FY.is_permutation([1, 2, 3], [1, 2, 2]))
check("is_single_cycle on a two-cycle-plus-fixed is False", not FY.is_single_cycle([1, 0, 2]))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall fisher_yates tests passed")

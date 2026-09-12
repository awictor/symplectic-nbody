"""Tests for permutation_test: exact enumeration, agreement with t-test, null calibration."""

import math
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from permutation_test import (permutation_test, sign_flip_test, diff_of_means,  # noqa: E402
                              diff_of_medians, t_statistic, mean, median, _LCG)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def brute_exact_p(a, b, statistic, alternative="two-sided"):
    """Independent reference: full enumeration of the two-sample permutation p-value."""
    pool = a + b
    n = len(pool)
    na = len(a)
    observed = statistic(a, b)
    count = 0
    total = 0
    for idx in combinations(range(n), na):
        idx_set = set(idx)
        ga = [pool[i] for i in idx]
        gb = [pool[i] for i in range(n) if i not in idx_set]
        s = statistic(ga, gb)
        if alternative == "two-sided":
            hit = abs(s) >= abs(observed) - 1e-12
        elif alternative == "greater":
            hit = s >= observed - 1e-12
        else:
            hit = s <= observed + 1e-12
        count += hit
        total += 1
    return count / total


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def ttest_p_two_sided(a, b):
    """Welch t-test p-value using a normal approximation to the t-distribution (large n)."""
    t = t_statistic(a, b)
    return 2 * (1 - normal_cdf(abs(t)))


def main():
    # ---- 1. exact path matches an independent brute-force enumeration -----------------
    a = [5.1, 6.2, 4.8, 7.0, 5.5]
    b = [3.2, 4.1, 2.9, 3.8]
    r = permutation_test(a, b, diff_of_means, exact_threshold=100000)
    ref = brute_exact_p(a, b, diff_of_means)
    check("exact diff_of_means matches brute force", r["exact"] and abs(r["p_value"] - ref) < 1e-12,
          f"{r['p_value']} vs {ref}")
    check("exact n_permutations == C(9,5)", r["n_permutations"] == math.comb(9, 5))

    # exact with median statistic
    rm = permutation_test(a, b, diff_of_medians, exact_threshold=100000)
    refm = brute_exact_p(a, b, diff_of_medians)
    check("exact diff_of_medians matches brute force", abs(rm["p_value"] - refm) < 1e-12)

    # one-sided alternatives
    rg = permutation_test(a, b, diff_of_means, alternative="greater", exact_threshold=100000)
    refg = brute_exact_p(a, b, diff_of_means, "greater")
    check("exact greater matches brute force", abs(rg["p_value"] - refg) < 1e-12)
    rl = permutation_test(a, b, diff_of_means, alternative="less", exact_threshold=100000)
    refl = brute_exact_p(a, b, diff_of_means, "less")
    check("exact less matches brute force", abs(rl["p_value"] - refl) < 1e-12)
    # greater + less tails should sum to slightly over 1 (the observed point counted in both)
    check("one-sided tails cover the mass", rg["p_value"] + rl["p_value"] >= 1.0 - 1e-9)

    # a clearly-separated pair should be significant; overlapping should not
    hi = [10.0, 11.0, 12.0, 13.0]
    lo = [1.0, 2.0, 3.0, 4.0]
    rsig = permutation_test(hi, lo, diff_of_means, exact_threshold=100000)
    check("separated groups give smallest possible exact p", rsig["p_value"] <= 2.0 / math.comb(8, 4) + 1e-12,
          f"p={rsig['p_value']}")

    # ---- 2. Monte-Carlo path tracks the exact enumeration ------------------------------
    # force MC by lowering the threshold below C(9,5)=126
    rmc = permutation_test(a, b, diff_of_means, n_resamples=40000, seed=1, exact_threshold=10)
    check("MC flagged non-exact", not rmc["exact"])
    check("MC p within sampling error of exact", abs(rmc["p_value"] - ref) < 0.02,
          f"mc={rmc['p_value']} exact={ref}")
    check("MC p is never zero", rmc["p_value"] > 0.0)

    # reproducibility: same seed -> identical p
    r1 = permutation_test(a, b, diff_of_means, n_resamples=5000, seed=99, exact_threshold=10)
    r2 = permutation_test(a, b, diff_of_means, n_resamples=5000, seed=99, exact_threshold=10)
    check("MC reproducible with same seed", r1["p_value"] == r2["p_value"])

    # ---- 3. agreement with the analytic t-test on large normal samples -----------------
    rng = _LCG(2024)

    def gauss(mu, sigma):
        u = sum((rng._next() >> 8) / (1 << 24) for _ in range(12)) - 6.0
        return mu + sigma * u

    ga = [gauss(0.0, 1.0) for _ in range(60)]
    gb = [gauss(0.8, 1.0) for _ in range(60)]
    rt = permutation_test(ga, gb, t_statistic, n_resamples=20000, seed=7, exact_threshold=10)
    pt = ttest_p_two_sided(ga, gb)
    check("permutation p tracks t-test p on normal data", abs(rt["p_value"] - pt) < 0.03,
          f"perm={rt['p_value']:.4f} t={pt:.4f}")

    # ---- 4. null calibration: same distribution -> uniform p, ~alpha rejection ---------
    rejections = 0
    trials = 150
    alpha = 0.10
    crng = _LCG(555)

    def gauss2(mu, sigma):
        u = sum((crng._next() >> 8) / (1 << 24) for _ in range(12)) - 6.0
        return mu + sigma * u

    for t in range(trials):
        x = [gauss2(5.0, 2.0) for _ in range(12)]
        y = [gauss2(5.0, 2.0) for _ in range(12)]
        res = permutation_test(x, y, diff_of_means, n_resamples=800, seed=1000 + t, exact_threshold=10)
        if res["p_value"] < alpha:
            rejections += 1
    rate = rejections / trials
    check("null rejection rate near alpha", rate < 0.22, f"rate={rate:.3f} (alpha={alpha})")

    # ---- 5. sign-flip paired test -----------------------------------------------------
    # exact enumeration for small n vs brute force
    d = [1.2, -0.3, 0.8, 2.1, 0.5, -0.1]
    sr = sign_flip_test(d, exact_threshold=20)
    # brute reference
    n = len(d)
    obs = sum(d) / n
    cnt = 0
    for mask in range(1 << n):
        s = sum(d[i] if (mask >> i) & 1 else -d[i] for i in range(n)) / n
        cnt += abs(s) >= abs(obs) - 1e-12
    check("sign-flip exact matches brute force", sr["exact"] and abs(sr["p_value"] - cnt / (1 << n)) < 1e-12,
          f"{sr['p_value']} vs {cnt/(1<<n)}")

    # a strong positive shift should be significant
    dpos = [2.0, 2.5, 1.8, 2.2, 3.0, 2.1, 1.9, 2.4]
    srp = sign_flip_test(dpos, exact_threshold=20)
    check("sign-flip detects consistent positive shift", srp["p_value"] < 0.05, f"p={srp['p_value']}")

    # sign-flip MC path reproducible and near exact
    dbig = [0.4 * (i % 5 - 2) + 0.3 for i in range(25)]
    se = sign_flip_test(dbig, n_resamples=20000, seed=3, exact_threshold=5)
    check("sign-flip MC non-exact and valid p", (not se["exact"]) and 0 < se["p_value"] <= 1)
    se2 = sign_flip_test(dbig, n_resamples=20000, seed=3, exact_threshold=5)
    check("sign-flip MC reproducible", se["p_value"] == se2["p_value"])

    # ---- 6. input validation ----------------------------------------------------------
    try:
        permutation_test([], [1, 2], diff_of_means)
        check("empty group raises", False)
    except ValueError:
        check("empty group raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

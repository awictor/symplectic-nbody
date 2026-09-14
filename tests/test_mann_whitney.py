"""Validate Mann-Whitney: identical groups, separation, exact vs normal, effect size, rank invariance, ties."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import mann_whitney as mw


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def brute_effect(a, b):
    """Fraction of cross-pairs where a > b (ties = 1/2)."""
    wins = 0.0
    for x in a:
        for y in b:
            if x > y:
                wins += 1
            elif x == y:
                wins += 0.5
    return wins / (len(a) * len(b))


def main():
    print("Mann-Whitney tests")

    # --- identical groups: U1 = U2 = n1 n2 / 2, p ~ 1 ---
    a = [1, 2, 3, 4, 5, 6]
    b = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]   # interleaved, no ties, ~symmetric
    u1, u2, _r = mw.u_statistic(a, b)
    check("U1 + U2 == n1 n2", abs(u1 + u2 - len(a) * len(b)) < 1e-9)
    res = mw.mann_whitney([10, 20, 30, 40], [15, 25, 35, 45])
    check(f"balanced groups -> high p ({res['p_value']:.2f})", res["p_value"] > 0.5)

    # --- cleanly separated groups: U collapses, tiny p ---
    lo = [1, 2, 3, 4, 5]
    hi = [101, 102, 103, 104, 105]
    res = mw.mann_whitney(lo, hi)
    check("separated groups: U == 0", res["U"] == 0)
    check(f"separated groups: tiny p ({res['p_value']:.4f})", res["p_value"] < 0.01)
    check("separated groups: effect size 0 (a all below b)", abs(res["effect_size"]) < 1e-9)

    # --- effect size == brute-force cross-pair fraction ---
    import random
    a = [3.1, 5.2, 1.0, 8.8, 4.4, 2.2, 7.7]
    b = [6.6, 0.5, 9.9, 3.3, 5.5]
    check(f"effect size matches brute force ({mw.effect_size(a, b):.4f})",
          abs(mw.effect_size(a, b) - brute_effect(a, b)) < 1e-9)

    # --- exact and normal-approx p-values agree for moderate n (no ties) ---
    a = [12, 15, 21, 7, 9, 30, 18]
    b = [20, 25, 11, 33, 40, 8, 27]
    pe = mw.exact_pvalue(a, b)
    pn = mw.normal_pvalue(a, b)
    check(f"exact ~ normal-approx ({pe:.3f} vs {pn:.3f})", abs(pe - pn) < 0.15)

    # --- rank invariance: monotone transform doesn't change U or p ---
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.5, 3.5, 4.5, 5.5, 6.5]
    r1 = mw.mann_whitney(a, b)
    a2 = [math.exp(x) for x in a]   # monotone transform
    b2 = [math.exp(x) for x in b]
    r2 = mw.mann_whitney(a2, b2)
    check("U invariant under monotone transform", r1["U"] == r2["U"])
    check("p invariant under monotone transform", abs(r1["p_value"] - r2["p_value"]) < 1e-9)

    # --- ties handled: average ranks, uses normal approx with tie correction ---
    a = [1, 2, 2, 3, 3]
    b = [2, 3, 3, 4, 4]
    res = mw.mann_whitney(a, b)
    check("tied data uses normal approx", res["method"] == "normal-approx")
    check("tied U1 + U2 == n1 n2", abs(res["U1"] + res["U2"] - 25) < 1e-9)

    # --- exact null distribution sums to C(n1+n2, n1) ---
    from math import comb
    counts = mw._exact_null_counts(4, 5)
    check("exact null counts sum to C(9,4)", sum(counts) == comb(9, 4))
    # symmetric distribution
    check("exact null symmetric", counts == counts[::-1])

    # --- one-sided tests ---
    lo = [1, 2, 3, 4, 5]
    hi = [6, 7, 8, 9, 10]
    p_less = mw.mann_whitney(lo, hi, alternative="less")["p_value"]
    p_greater = mw.mann_whitney(lo, hi, alternative="greater")["p_value"]
    check(f"a<b: 'less' p small ({p_less:.4f})", p_less < 0.05)
    check(f"a<b: 'greater' p large ({p_greater:.4f})", p_greater > 0.95)

    # --- average ranks correct on a hand case ---
    ranks = mw._average_ranks([10, 20, 20, 30])
    check("average ranks for ties", ranks == [1.0, 2.5, 2.5, 4.0])

    # --- deterministic ---
    a = [5, 3, 8, 1]
    b = [4, 9, 2, 7]
    check("deterministic", mw.mann_whitney(a, b) == mw.mann_whitney(a, b))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

"""Tests for SOS DP: zeta/moebius inverse pair, subset/superset transforms, OR/AND/subset-sum conv."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sos_dp import (  # noqa: E402
    zeta_subset,
    moebius_subset,
    zeta_superset,
    moebius_superset,
    or_convolution,
    and_convolution,
    subset_sum_convolution,
    brute_zeta_subset,
    brute_or_convolution,
    brute_and_convolution,
    brute_subset_sum_convolution,
)


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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def _rand_arr(size, rng, lo=-5, hi=5):
    return [(rng() % (hi - lo + 1)) + lo for _ in range(size)]


def _brute_superset(f):
    size = len(f)
    G = [0] * size
    for S in range(size):
        for T in range(size):
            if (S & T) == S:  # T superset of S
                G[S] += f[T]
    return G


def main():
    # ---- 1. zeta_subset matches O(3^n) brute definition ---------------------------------
    rng = _lcg(2024)
    mism = 0
    for _ in range(200):
        n = rng() % 6  # 0..5 bits
        f = _rand_arr(1 << n, rng)
        if zeta_subset(f) != brute_zeta_subset(f):
            mism += 1
    check("zeta_subset == O(3^n) brute definition (200 arrays)", mism == 0, f"{mism}")

    # ---- 2. zeta and moebius are exact inverses -----------------------------------------
    rng = _lcg(77)
    ok_sub = ok_sup = True
    for _ in range(200):
        n = rng() % 6
        f = _rand_arr(1 << n, rng)
        if moebius_subset(zeta_subset(f)) != f:
            ok_sub = False
        if moebius_superset(zeta_superset(f)) != f:
            ok_sup = False
    check("moebius_subset inverts zeta_subset", ok_sub)
    check("moebius_superset inverts zeta_superset", ok_sup)

    # ---- 3. superset transform matches brute --------------------------------------------
    rng = _lcg(7)
    mism = 0
    for _ in range(150):
        n = rng() % 6
        f = _rand_arr(1 << n, rng)
        if zeta_superset(f) != _brute_superset(f):
            mism += 1
    check("zeta_superset == brute superset sum", mism == 0, f"{mism}")

    # ---- 4. OR / AND / subset-sum convolutions match their defining sums ----------------
    rng = _lcg(555)
    mism_or = mism_and = mism_ss = 0
    for _ in range(200):
        n = 1 + rng() % 5  # 1..5 bits
        f = _rand_arr(1 << n, rng)
        g = _rand_arr(1 << n, rng)
        if or_convolution(f, g) != brute_or_convolution(f, g):
            mism_or += 1
        if and_convolution(f, g) != brute_and_convolution(f, g):
            mism_and += 1
        if subset_sum_convolution(f, g) != brute_subset_sum_convolution(f, g):
            mism_ss += 1
    check("OR-convolution == brute (200 pairs)", mism_or == 0, f"{mism_or}")
    check("AND-convolution == brute (200 pairs)", mism_and == 0, f"{mism_and}")
    check("subset-sum convolution == brute (200 pairs)", mism_ss == 0, f"{mism_ss}")

    # ---- 5. hand examples ---------------------------------------------------------------
    # n=2, f indexed by masks {00,01,10,11}. f = [1,1,1,1]
    # zeta: F(S) = number of subsets of S = 2^popcount(S) -> [1,2,2,4]
    check("zeta of all-ones = 2^popcount", zeta_subset([1, 1, 1, 1]) == [1, 2, 2, 4])
    # moebius of that returns all ones
    check("moebius round-trip", moebius_subset([1, 2, 2, 4]) == [1, 1, 1, 1])

    # OR-convolution of indicator{00} with anything is identity (00 is OR-identity)
    delta0 = [1, 0, 0, 0]
    g = [3, 5, 7, 9]
    check("OR-conv with delta_emptyset is identity", or_convolution(delta0, g) == g)
    # AND-convolution with all-ones-mask indicator: AND with full set is identity
    delta_full = [0, 0, 0, 1]  # mask 11 = full universe
    check("AND-conv with delta_fullset is identity", and_convolution(delta_full, g) == g)

    # subset-sum conv: f=g=delta at singletons {0} and {1}: [0,1,0,0] * [0,0,1,0]
    # only disjoint pair (A={0}, B={1}) -> S={0,1}=mask 3
    a = [0, 1, 0, 0]  # mask 01
    b = [0, 0, 1, 0]  # mask 10
    ss = subset_sum_convolution(a, b)
    check("subset-sum conv of two disjoint singletons -> mask 3", ss == [0, 0, 0, 1])

    # ---- 6. edge cases ------------------------------------------------------------------
    check("n=0 (single element) zeta identity", zeta_subset([5]) == [5])
    check("n=0 moebius identity", moebius_subset([5]) == [5])
    try:
        zeta_subset([1, 2, 3])  # not a power of two
        check("non-power-of-two raises", False)
    except ValueError:
        check("non-power-of-two raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

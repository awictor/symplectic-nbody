"""Validate the ballot problem: formula vs reflection vs cycle-lemma vs brute force, plus Catalan numbers."""

import math
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ballot_problem as bp


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Bertrand ballot problem tests")

    # --- Bertrand's formula on a known case: p=3, q=2 -> (3-2)/(3+2) = 1/5 ---
    check("P(3,2) == 1/5", bp.ballot_probability(3, 2) == Fraction(1, 5))
    check("P(5,2) == 3/7", bp.ballot_probability(5, 2) == Fraction(3, 7))
    check("landslide P(10,0) == 1", bp.ballot_probability(10, 0) == Fraction(1))
    check("tie-margin p<=q gives 0", bp.ballot_probability(3, 3) == 0 and bp.ballot_probability(2, 5) == 0)

    # --- probability == strict_count / total arrangements ---
    for p, q in [(3, 2), (5, 3), (6, 4), (7, 2)]:
        total = math.comb(p + q, p)
        prob = Fraction(bp.ballot_count_strict(p, q), total)
        check(f"count/total == formula at ({p},{q})", prob == bp.ballot_probability(p, q))

    # --- reflection principle and cycle lemma agree with the closed form ---
    for p in range(1, 9):
        for q in range(0, p):
            c1 = bp.ballot_count_strict(p, q)
            c2 = bp.ballot_count_strict_reflection(p, q)
            c3 = bp.ballot_count_cycle_lemma(p, q)
            if not (c1 == c2 == c3):
                check(f"reflection/cycle agree at ({p},{q}): {c1},{c2},{c3}", False)
    check("reflection & cycle-lemma match closed form for all p<=8", _failed == 0)

    # --- brute force confirms the strict count for small p, q ---
    bad = 0
    for p in range(1, 8):
        for q in range(0, p + 2):  # include p<=q where the count is 0
            if bp.ballot_count_strict(p, q) != bp.brute_force_strict(p, q):
                bad += 1
    check(f"brute-force strict count matches formula ({bad} mismatches)", bad == 0)

    # --- weak ballot count vs brute force (ties allowed) ---
    badw = 0
    for p in range(0, 8):
        for q in range(0, p + 1):  # weak needs p >= q
            if bp.ballot_count_weak(p, q) != bp.brute_force_weak(p, q):
                badw += 1
    check(f"brute-force weak count matches ballot number ({badw} mismatches)", badw == 0)

    # --- Catalan numbers: known sequence and weak count at p=q ---
    known = [1, 1, 2, 5, 14, 42, 132, 429, 1430]
    check("catalan sequence matches", [bp.catalan(n) for n in range(len(known))] == known)
    check("catalan(n) == weak(n,n)", all(bp.catalan(n) == bp.ballot_count_weak(n, n) for n in range(1, 8)))
    # Dyck paths: weak count at p=q=n brute-forced equals catalan(n)
    check("Dyck-path brute force == catalan", all(bp.brute_force_weak(n, n) == bp.catalan(n) for n in range(0, 7)))

    # --- strict count first vote must be A: strict(p,q) == weak(p-1,q) mapped ---
    # a strict-lead sequence is an A followed by a weak sequence on (p-1, q) that never drops below +1,
    # equivalently a weak (A>=B) sequence on the remaining p-1 A's and q B's. Check the identity.
    for p in range(1, 8):
        for q in range(0, p):
            check2 = bp.ballot_count_strict(p, q) == bp.ballot_count_weak(p - 1, q)
            if not check2:
                check(f"strict(p,q)==weak(p-1,q) at ({p},{q})", False)
    check("strict(p,q) == weak(p-1,q) identity holds", _failed == 0)

    # --- monotonicity: bigger margin -> higher lead-throughout probability ---
    probs = [bp.ballot_probability(p, 3) for p in range(4, 10)]
    check("probability rises with margin", all(probs[i] < probs[i + 1] for i in range(len(probs) - 1)))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

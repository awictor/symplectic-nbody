"""Validate CRP: expected tables formula, alpha ln n growth, EPPF exchangeability, concentration limits."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import chinese_restaurant as crp


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Chinese restaurant process tests")

    # --- mean tables matches the harmonic-sum formula ---
    for n, alpha in [(50, 1.0), (100, 2.0), (30, 0.5)]:
        emp = crp.mean_tables(n, alpha, n_runs=3000, seed=1)
        theo = crp.expected_tables(n, alpha)
        check(f"E[K] ~ formula (n={n},a={alpha}: {emp:.2f} vs {theo:.2f})", abs(emp - theo) < 0.5)

    # --- expected tables grows like alpha ln n ---
    alpha = 2.0
    k100 = crp.expected_tables(100, alpha)
    k1000 = crp.expected_tables(1000, alpha)
    # difference ~ alpha ln(1000/100) = alpha ln 10
    check(f"K grows like alpha ln n ({k1000 - k100:.2f} ~ {alpha * math.log(10):.2f})",
          abs((k1000 - k100) - alpha * math.log(10)) < 0.3)

    # --- larger alpha -> more tables ---
    k_small = crp.expected_tables(100, 0.5)
    k_large = crp.expected_tables(100, 5.0)
    check(f"larger alpha -> more tables ({k_small:.1f} < {k_large:.1f})", k_small < k_large)

    # --- EPPF matches the sequential seating probability ---
    # partition of 5 customers into blocks {3,2}
    alpha = 1.5
    # a specific seating giving sizes [3,2]: table0 x3, table1 x2 in some order
    assignment = [0, 0, 1, 0, 1]
    sizes = [3, 2]
    p_seq = crp.sequential_probability(assignment, alpha)
    p_eppf = crp.partition_probability(sizes, alpha)
    # the EPPF counts ALL orderings giving this partition; a single ordering is one of them.
    # Instead, sum sequential probs over all orderings and compare to EPPF * (number of orderings)?
    # Simpler: EPPF should equal the sum of sequential probs over all labelled seatings with these blocks.
    # Verify exchangeability directly below; here check a single seating <= EPPF.
    check(f"single seating prob positive and <= EPPF ({p_seq:.5f} <= {p_eppf:.5f})", 0 < p_seq <= p_eppf + 1e-12)

    # --- exchangeability: reordering customers gives the same partition probability ---
    # two different orderings that yield the SAME partition {3,2} should have EQUAL sequential prob
    a1 = [0, 0, 0, 1, 1]   # first three together, last two together
    a2 = [0, 1, 0, 1, 0]   # interleaved but same block sizes {3,2}
    check(f"exchangeable seatings equal ({crp.sequential_probability(a1, alpha):.6f} == {crp.sequential_probability(a2, alpha):.6f})",
          abs(crp.sequential_probability(a1, alpha) - crp.sequential_probability(a2, alpha)) < 1e-12)

    # --- EPPF equals the shared sequential probability of ANY seating with those block sizes ---
    # By exchangeability, every ordered seating giving blocks {3,2} has the same sequential probability,
    # and the EPPF (the probability of that unlabeled partition-of-sizes) equals that shared value.
    p_common = crp.sequential_probability(a1, alpha)
    check(f"EPPF == shared seating prob ({p_eppf:.5f} vs {p_common:.5f})",
          abs(p_eppf - p_common) < 1e-9)

    # --- partition probabilities sum to 1 over all partitions of a small n ---
    # enumerate all set partitions of {0,1,2} and sum EPPF
    def set_partitions(elements):
        if not elements:
            yield []
            return
        first = elements[0]
        for rest in set_partitions(elements[1:]):
            # add first to each existing block
            for i in range(len(rest)):
                yield rest[:i] + [[first] + rest[i]] + rest[i + 1:]
            # or as its own block
            yield [[first]] + rest
    alpha = 2.0
    total = 0.0
    for part in set_partitions([0, 1, 2]):
        sizes = [len(b) for b in part]
        total += crp.partition_probability(sizes, alpha)
    check(f"partition probabilities sum to 1 (n=3): {total:.6f}", abs(total - 1.0) < 1e-9)

    # --- concentration limits: tiny alpha -> few tables, big alpha -> many ---
    _a, tables_small = crp.simulate(50, 0.05, seed=1)
    _a, tables_big = crp.simulate(50, 50.0, seed=1)
    check(f"tiny alpha -> few tables ({len(tables_small)})", len(tables_small) <= 3)
    check(f"big alpha -> many tables ({len(tables_big)})", len(tables_big) > 20)

    # --- table sizes sum to n ---
    _a, tables = crp.simulate(100, 1.0, seed=3)
    check("table sizes sum to n", sum(tables) == 100)

    # --- first customer always starts table 0 ---
    assign, _ = crp.simulate(10, 1.0, seed=1)
    check("first customer at table 0", assign[0] == 0)

    # --- expected_tables(n=1) == 1 (first customer always sits) ---
    check("E[K_1] == 1", abs(crp.expected_tables(1, 3.0) - 1.0) < 1e-12)

    # --- deterministic ---
    check("deterministic", crp.simulate(50, 1.5, seed=42) == crp.simulate(50, 1.5, seed=42))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

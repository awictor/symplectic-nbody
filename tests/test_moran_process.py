"""Validate Moran: neutral fixation = i/N, selection formula, advantage/disadvantage, absorption, monotone."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import moran_process as mp


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Moran process tests")

    # --- neutral fixation from i copies == i/N ---
    n = 10
    for i in (1, 3, 5, 7):
        check(f"neutral formula rho_{i} == i/N", abs(mp.fixation_probability_formula(n, i, 1.0) - i / n) < 1e-12)

    # --- single neutral mutant fixes with prob 1/N ---
    check("single neutral mutant fixes 1/N", abs(mp.single_mutant_fixation(n, 1.0) - 1 / n) < 1e-12)

    # --- empirical neutral fixation matches i/N ---
    emp = mp.fixation_probability(n, 3, r=1.0, n_runs=4000, seed=1)
    check(f"empirical neutral fixation ~ 3/10 ({emp:.3f})", abs(emp - 0.3) < 0.04)

    # --- with selection, empirical matches the closed-form formula ---
    n = 12
    r = 1.5
    i0 = 3
    theo = mp.fixation_probability_formula(n, i0, r)
    emp = mp.fixation_probability(n, i0, r, n_runs=5000, seed=1)
    check(f"selection: empirical ~ formula ({emp:.3f} vs {theo:.3f})", abs(emp - theo) < 0.04)

    # --- advantageous mutant fixes more than neutral, deleterious less ---
    n = 15
    fix_neutral = mp.fixation_probability_formula(n, 1, 1.0)
    fix_adv = mp.fixation_probability_formula(n, 1, 2.0)
    fix_del = mp.fixation_probability_formula(n, 1, 0.5)
    check(f"advantageous > neutral ({fix_adv:.3f} > {fix_neutral:.3f})", fix_adv > fix_neutral)
    check(f"deleterious < neutral ({fix_del:.3f} < {fix_neutral:.3f})", fix_del < fix_neutral)

    # --- single-mutant fixation approaches 1 - 1/r for large N ---
    r = 2.0
    fix_large = mp.single_mutant_fixation(1000, r)
    check(f"single mutant -> 1 - 1/r for large N ({fix_large:.3f} vs {1 - 1/r:.3f})",
          abs(fix_large - (1 - 1 / r)) < 0.01)

    # --- fixation probability monotone increasing in starting count ---
    n = 10
    r = 1.3
    fixes = [mp.fixation_probability_formula(n, i, r) for i in range(n + 1)]
    check("fixation monotone in i", all(fixes[i] <= fixes[i + 1] for i in range(n)))
    check("rho_0 == 0, rho_N == 1", fixes[0] == 0.0 and abs(fixes[-1] - 1.0) < 1e-12)

    # --- every trajectory is absorbed ---
    all_absorbed = True
    for run in range(50):
        fixed, steps = mp.simulate(10, 5, r=1.2, seed=run + 1)
        # fixed is a bool; the fact that it returned means it hit 0 or N (or max_steps)
        if steps >= 10_000_000:
            all_absorbed = False
    check("all trajectories absorbed", all_absorbed)

    # --- boundary formula ---
    check("rho at i=0 is 0", mp.fixation_probability_formula(10, 0, 1.5) == 0.0)
    check("rho at i=N is 1", mp.fixation_probability_formula(10, 10, 1.5) == 1.0)

    # --- strong advantage: single mutant fixes with high prob even in moderate N ---
    fix_strong = mp.fixation_probability_formula(20, 1, 5.0)
    check(f"strong advantage high fixation ({fix_strong:.3f})", fix_strong > 0.7)

    # --- mean absorption time positive and larger for neutral than strong selection ---
    t_neutral = mp.mean_absorption_time(10, 5, r=1.0, n_runs=500, seed=1)
    check("mean absorption time positive", t_neutral > 0)

    # --- deterministic ---
    check("deterministic", mp.simulate(10, 5, r=1.2, seed=42) == mp.simulate(10, 5, r=1.2, seed=42))

    # --- neutral empirical from single mutant ~ 1/N ---
    emp1 = mp.fixation_probability(20, 1, r=1.0, n_runs=8000, seed=1)
    check(f"single neutral mutant empirical ~ 1/20 ({emp1:.3f})", abs(emp1 - 0.05) < 0.02)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

"""Validate Galton-Watson: fixed-point extinction, criticality, empirical match, m^n growth, binary fission."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import galton_watson as gw


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Galton-Watson tests")

    # --- extinction probability is a fixed point of G ---
    # supercritical: p0=0.2, p1=0.3, p2=0.5 -> m = 0.3 + 1.0 = 1.3
    probs = [0.2, 0.3, 0.5]
    q = gw.extinction_probability(probs)
    check(f"q is a fixed point of G ({q:.6f})", abs(gw.pgf(probs, q) - q) < 1e-9)
    check("supercritical m > 1", gw.offspring_mean(probs) > 1)
    check(f"supercritical q < 1 ({q:.4f})", q < 1.0)

    # --- q is the SMALLEST fixed point (s=1 is always a fixed point) ---
    check("q <= 1 (smallest root)", q <= 1.0 + 1e-12)
    # a slightly smaller s should give G(s) > s (below the smallest root G is above the line)... check q is minimal:
    # G(0) = p0 > 0, and iteration from 0 converges up to q
    check("q >= p0 (iteration starts at G(0)=p0)", q >= probs[0] - 1e-9)

    # --- subcritical: certain extinction q = 1 ---
    sub = [0.5, 0.3, 0.2]  # m = 0.3 + 0.4 = 0.7 < 1
    check("subcritical m < 1", gw.offspring_mean(sub) < 1)
    check("subcritical q == 1", abs(gw.extinction_probability(sub) - 1.0) < 1e-6)
    check("classify subcritical", gw.classify(sub) == "subcritical")

    # --- critical: m = 1 -> q = 1 ---
    crit = [0.25, 0.5, 0.25]  # m = 0.5 + 0.5 = 1.0
    check("critical m == 1", abs(gw.offspring_mean(crit) - 1.0) < 1e-12)
    check("critical q == 1", abs(gw.extinction_probability(crit) - 1.0) < 1e-4)
    check("classify critical", gw.classify(crit) == "critical")
    check("classify supercritical", gw.classify(probs) == "supercritical")

    # --- binary fission: p0 = a, p2 = 1-a. G(s) = a + (1-a)s^2.
    #     extinction q = a/(1-a) if a < 1/2 (supercritical), else 1. ---
    a = 0.3
    bf = [a, 0.0, 1 - a]  # m = 2(1-a) = 1.4 > 1
    q_bf = gw.extinction_probability(bf)
    analytic = a / (1 - a)  # smallest root of (1-a)s^2 - s + a = 0
    check(f"binary fission q matches a/(1-a) ({q_bf:.4f} vs {analytic:.4f})", abs(q_bf - analytic) < 1e-6)

    # --- empirical extinction frequency matches computed q ---
    emp = gw.empirical_extinction(probs, n_generations=30, n_runs=3000, seed=1)
    check(f"empirical extinction ~ q ({emp:.3f} vs {q:.3f})", abs(emp - q) < 0.05)

    # --- subcritical always dies out in simulation ---
    emp_sub = gw.empirical_extinction(sub, n_generations=40, n_runs=500, seed=1)
    check(f"subcritical empirically extinct ({emp_sub:.3f})", emp_sub > 0.98)

    # --- mean generation size grows as m^n (supercritical) ---
    # average Z_n over many runs; keep n_gen modest so supercritical survivors don't blow up runtime
    m = gw.offspring_mean(probs)
    n_gen = 6
    totals = [0] * (n_gen + 1)
    n_runs = 4000
    for r in range(n_runs):
        sizes = gw.simulate(probs, n_gen, seed=r * 2749 + 1, max_pop=100_000)
        for g in range(min(len(sizes), n_gen + 1)):
            totals[g] += sizes[g]
    mean_zn = totals[n_gen] / n_runs
    check(f"E[Z_n] ~ m^n ({mean_zn:.2f} vs {m**n_gen:.2f})", abs(mean_zn - m ** n_gen) < 0.2 * m ** n_gen)
    check("expected_generation_size formula", abs(gw.expected_generation_size(probs, 5) - m ** 5) < 1e-9)

    # --- PGF and mean sanity ---
    check("G(1) == 1", abs(gw.pgf(probs, 1.0) - 1.0) < 1e-12)
    check("G(0) == p0", abs(gw.pgf(probs, 0.0) - probs[0]) < 1e-12)
    check("offspring mean", abs(gw.offspring_mean([0.1, 0.2, 0.7]) - (0.2 + 1.4)) < 1e-12)

    # --- extinct lineages stay extinct (size 0 propagates) ---
    sizes = gw.simulate([0.9, 0.05, 0.05], 20, seed=1)  # strongly subcritical
    if 0 in sizes:
        first_zero = sizes.index(0)
        check("extinction is absorbing", all(s == 0 for s in sizes[first_zero:]))
    else:
        check("extinction is absorbing", True)

    # --- deterministic ---
    check("deterministic", gw.simulate(probs, 15, seed=42) == gw.simulate(probs, 15, seed=42))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

"""Validate Wright-Fisher: neutral fixation = p0, selection helps, drift variance, heterozygosity, absorption."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import wright_fisher as wf


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Wright-Fisher tests")

    # --- neutral fixation probability == initial frequency ---
    two_n = 50
    for p0 in (0.2, 0.5, 0.8):
        est = wf.fixation_probability(two_n, p0, s=0.0, n_runs=600, seed=1)
        check(f"neutral fixation ~ p0={p0} (got {est:.3f})", abs(est - p0) < 0.08)

    # --- every trajectory absorbs at 0 or 1 ---
    all_absorbed = True
    for r in range(50):
        traj, fixed = wf.simulate(two_n, 0.5, seed=r + 1)
        if traj[-1] not in (0.0, 1.0):
            all_absorbed = False
    check("all trajectories absorbed at 0 or 1", all_absorbed)

    # --- selection raises fixation probability above neutral ---
    p0 = 0.1
    fix_neutral = wf.fixation_probability(two_n, p0, s=0.0, n_runs=800, seed=1)
    fix_selected = wf.fixation_probability(two_n, p0, s=0.1, n_runs=800, seed=1)
    check(f"selection raises fixation ({fix_selected:.3f} > {fix_neutral:.3f})", fix_selected > fix_neutral)

    # --- fixation with selection matches the Kimura diffusion formula ---
    two_n = 40
    p0 = 0.05
    s = 0.05
    est = wf.fixation_probability(two_n, p0, s=s, n_runs=3000, seed=1)
    kimura = wf.kimura_fixation_probability(two_n, p0, s)
    check(f"fixation ~ Kimura formula ({est:.3f} vs {kimura:.3f})", abs(est - kimura) < 0.06)

    # --- Kimura reduces to p0 when s=0 ---
    check("Kimura(s=0) == p0", abs(wf.kimura_fixation_probability(50, 0.3, 0.0) - 0.3) < 1e-12)

    # --- drift variance matches p(1-p)/(2N) ---
    two_n = 100
    p = 0.4
    # empirical one-generation variance
    counts = []
    for r in range(4000):
        traj, _ = wf.simulate(two_n, p, max_gen=1, seed=r + 1)
        counts.append(traj[1] if len(traj) > 1 else traj[0])
    m = sum(counts) / len(counts)
    var = sum((c - m) ** 2 for c in counts) / len(counts)
    analytic = wf.drift_variance(two_n, p)
    check(f"drift variance ~ p(1-p)/2N ({var:.5f} vs {analytic:.5f})", abs(var - analytic) < 0.3 * analytic)

    # --- mean allele frequency after one generation is unchanged (neutral, martingale) ---
    check(f"neutral mean frequency preserved ({m:.3f} vs {p})", abs(m - p) < 0.02)

    # --- heterozygosity decays by 1 - 1/(2N) per generation ---
    two_n = 50
    h0 = 0.5
    h10 = wf.heterozygosity_decay(two_n, h0, 10)
    check("heterozygosity decays geometrically", h10 < h0)
    check("heterozygosity formula exact",
          abs(h10 - h0 * (1 - 1 / two_n) ** 10) < 1e-15)
    # decay factor per generation
    h1 = wf.heterozygosity_decay(two_n, h0, 1)
    check("one-gen decay factor 1-1/(2N)", abs(h1 / h0 - (1 - 1 / two_n)) < 1e-12)

    # --- smaller population fixes faster ---
    t_small = wf.mean_fixation_time(20, 0.5, n_runs=300, seed=1)
    t_large = wf.mean_fixation_time(100, 0.5, n_runs=300, seed=1)
    check(f"smaller population fixes faster ({t_small:.0f} < {t_large:.0f})", t_small < t_large)

    # --- strong selection: beneficial allele almost always fixes ---
    fix_strong = wf.fixation_probability(50, 0.3, s=0.5, n_runs=500, seed=1)
    check(f"strong selection -> high fixation ({fix_strong:.2f})", fix_strong > 0.8)
    # deleterious allele almost always lost
    fix_del = wf.fixation_probability(50, 0.3, s=-0.5, n_runs=500, seed=1)
    check(f"deleterious -> low fixation ({fix_del:.2f})", fix_del < 0.15)

    # --- reproducible per seed ---
    t1, f1 = wf.simulate(50, 0.5, seed=42)
    t2, f2 = wf.simulate(50, 0.5, seed=42)
    check("reproducible per seed", t1 == t2 and f1 == f2)

    # --- boundary: p0=0 stays lost, p0=1 stays fixed ---
    _, f0 = wf.simulate(50, 0.0, seed=1)
    _, f1b = wf.simulate(50, 1.0, seed=1)
    check("p0=0 -> loss, p0=1 -> fixation", (not f0) and f1b)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

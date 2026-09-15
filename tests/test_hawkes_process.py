"""Validate Hawkes: Poisson limit, stationary rate mu/(1-n), intensity jumps, clustering, branching ratio."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import hawkes_process as hp


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    print("Hawkes process tests")

    # --- alpha=0 reduces to a Poisson process: count ~ mu * T ---
    mu = 2.0
    T = 500.0
    counts = []
    for r in range(20):
        ev = hp.simulate_fast(mu, 0.0, 1.0, T, seed=r + 1)
        counts.append(len(ev))
    check(f"Poisson limit (alpha=0): count ~ mu*T ({mean(counts):.0f} vs {mu*T:.0f})",
          abs(mean(counts) - mu * T) < 0.1 * mu * T)

    # --- empirical rate matches mu/(1-n) across branching ratios ---
    for alpha, beta in [(0.5, 2.0), (1.0, 2.0), (1.5, 2.0)]:  # n = 0.25, 0.5, 0.75
        n = hp.branching_ratio(alpha, beta)
        theo = hp.stationary_rate(mu, alpha, beta)
        rates = []
        for r in range(20):
            ev = hp.simulate_fast(mu, alpha, beta, T, seed=r + 100)
            rates.append(hp.empirical_rate(ev, T))
        emp = mean(rates)
        check(f"rate ~ mu/(1-n) for n={n:.2f} ({emp:.2f} vs {theo:.2f})", abs(emp - theo) < 0.15 * theo)

    # --- branching ratio and stationary formula ---
    check("branching ratio == alpha/beta", abs(hp.branching_ratio(1.5, 2.0) - 0.75) < 1e-12)
    check("stationary rate formula", abs(hp.stationary_rate(2.0, 1.0, 2.0) - 2.0 / 0.5) < 1e-12)
    check("supercritical rate infinite", hp.stationary_rate(1.0, 3.0, 2.0) == float("inf"))

    # --- intensity jumps by alpha at each event and decays ---
    events = [1.0, 2.0]
    mu2, a2, b2 = 1.0, 2.0, 1.0
    lam_before = hp.intensity(1.0 - 1e-9, events, mu2, a2, b2)
    lam_after = hp.intensity(1.0 + 1e-9, events, mu2, a2, b2)
    check(f"intensity jumps ~ alpha at event ({lam_after - lam_before:.3f} ~ {a2})",
          abs((lam_after - lam_before) - a2) < 0.01)
    # decay: intensity at t=1.5 (one event at t=1 active) = mu + alpha exp(-beta*0.5)
    lam_mid = hp.intensity(1.5, [1.0], mu2, a2, b2)
    check("intensity decays exponentially", abs(lam_mid - (mu2 + a2 * math.exp(-b2 * 0.5))) < 1e-9)

    # --- clustering: Hawkes inter-event gaps have higher variance than Poisson of same rate ---
    ev = hp.simulate_fast(1.0, 0.8, 1.0, 2000.0, seed=7)   # n=0.8, clustered
    gaps = [ev[i + 1] - ev[i] for i in range(len(ev) - 1)]
    m = mean(gaps)
    var = mean([(g - m) ** 2 for g in gaps])
    cv2 = var / (m * m)   # coefficient of variation squared; Poisson (exponential gaps) -> 1
    check(f"Hawkes gaps overdispersed vs Poisson (CV^2 {cv2:.2f} > 1)", cv2 > 1.2)

    # --- near-critical process produces far more events than baseline ---
    ev_base = hp.simulate_fast(1.0, 0.0, 2.0, 500.0, seed=3)
    ev_crit = hp.simulate_fast(1.0, 1.8, 2.0, 500.0, seed=3)   # n=0.9
    check(f"near-critical amplifies count ({len(ev_crit)} >> {len(ev_base)})",
          len(ev_crit) > 3 * len(ev_base))

    # --- events are sorted and within [0, T) ---
    ev = hp.simulate_fast(1.0, 0.5, 2.0, 100.0, seed=1)
    check("events sorted", all(ev[i] <= ev[i + 1] for i in range(len(ev) - 1)))
    check("events within [0, T)", all(0 <= e < 100.0 for e in ev))

    # --- simulate and simulate_fast agree statistically (same rate) ---
    r_slow = mean([hp.empirical_rate(hp.simulate(1.0, 1.0, 2.0, 300.0, seed=r + 1), 300.0) for r in range(15)])
    r_fast = mean([hp.empirical_rate(hp.simulate_fast(1.0, 1.0, 2.0, 300.0, seed=r + 1), 300.0) for r in range(15)])
    check(f"slow and fast simulators agree ({r_slow:.2f} vs {r_fast:.2f})", abs(r_slow - r_fast) < 0.2)

    # --- reproducible per seed ---
    a = hp.simulate_fast(1.0, 1.0, 2.0, 100.0, seed=42)
    b = hp.simulate_fast(1.0, 1.0, 2.0, 100.0, seed=42)
    check("reproducible per seed", a == b)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

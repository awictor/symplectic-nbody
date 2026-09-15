"""Validate coalescent: E[TMRCA], E[total length], n=2 exponential, segregating sites, Watterson estimator."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import coalescent as co


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def mean(xs):
    return sum(xs) / len(xs)


def main():
    print("Coalescent tests")

    # --- mean T_MRCA matches 2(1 - 1/n) ---
    for n in (2, 5, 10):
        tmrcas = [co.simulate(n, seed=r + 1)["t_mrca"] for r in range(3000)]
        emp = mean(tmrcas)
        theory = co.expected_t_mrca(n)
        check(f"E[T_MRCA] n={n} ~ 2(1-1/n) ({emp:.3f} vs {theory:.3f})", abs(emp - theory) < 0.1)

    # --- mean total branch length matches 2 H_{n-1} ---
    for n in (5, 10):
        lengths = [co.simulate(n, seed=r + 1)["total_length"] for r in range(3000)]
        emp = mean(lengths)
        theory = co.expected_total_length(n)
        check(f"E[total length] n={n} ~ 2 H_(n-1) ({emp:.3f} vs {theory:.3f})", abs(emp - theory) < 0.15)

    # --- n=2 coalescence time is exponential with mean 1 ---
    times = [co.simulate(2, seed=r + 1)["t_mrca"] for r in range(5000)]
    m = mean(times)
    var = sum((t - m) ** 2 for t in times) / len(times)
    check(f"n=2 mean ~ 1 ({m:.3f})", abs(m - 1.0) < 0.05)
    check(f"n=2 exponential (var ~ mean^2, {var:.3f})", abs(var - 1.0) < 0.15)

    # --- genealogy always reduces to one MRCA with n-1 coalescences ---
    for n in (3, 7, 12):
        g = co.simulate(n, seed=1)
        check(f"n={n}: n-1 coalescences", g["n_coalescences"] == n - 1)
        check(f"n={n}: {n-1} coalescence times", len(g["coalescence_times"]) == n - 1)

    # --- coalescence times shrink as more lineages remain (higher rate) ---
    # first time (k=n, highest rate) should on average be shorter than the last (k=2)
    n = 10
    first_times = []
    last_times = []
    for r in range(2000):
        ct = co.simulate(n, seed=r + 1)["coalescence_times"]
        first_times.append(ct[0])   # k=n
        last_times.append(ct[-1])   # k=2
    check(f"early coalescence faster than late ({mean(first_times):.3f} < {mean(last_times):.3f})",
          mean(first_times) < mean(last_times))

    # --- mean segregating sites matches theta H_{n-1} ---
    n = 8
    theta = 5.0
    Ss = [co.segregating_sites(n, theta, seed=r + 1)[0] for r in range(3000)]
    emp = mean(Ss)
    theory = co.expected_segregating_sites(n, theta)
    check(f"E[S] ~ theta H_(n-1) ({emp:.2f} vs {theory:.2f})", abs(emp - theory) < 0.5)

    # --- Watterson estimator recovers theta ---
    theta_hats = [co.watterson_theta(s, n) for s in Ss]
    check(f"Watterson theta_hat recovers theta ({mean(theta_hats):.2f} vs {theta})",
          abs(mean(theta_hats) - theta) < 0.3)

    # --- Watterson estimator formula: S / H_{n-1} ---
    check("Watterson formula", abs(co.watterson_theta(10, 5) - 10 / co.harmonic(4)) < 1e-12)

    # --- harmonic numbers ---
    check("harmonic(1) == 1", co.harmonic(1) == 1.0)
    check("harmonic(4) == 25/12", abs(co.harmonic(4) - 25 / 12) < 1e-12)

    # --- expected T_MRCA approaches 2 for large n ---
    check("E[T_MRCA] -> 2 for large n", abs(co.expected_t_mrca(1000) - 2.0) < 0.01)

    # --- total length is sum of k * time_k ---
    g = co.simulate(6, seed=3)
    ct = g["coalescence_times"]
    reconstructed = sum((6 - i) * ct[i] for i in range(len(ct)))
    check("total length = sum k*t_k", abs(g["total_length"] - reconstructed) < 1e-9)

    # --- deterministic ---
    check("deterministic", co.simulate(10, seed=42) == co.simulate(10, seed=42))

    # --- more mutations with larger theta ---
    s_lo = mean([co.segregating_sites(8, 2.0, seed=r + 1)[0] for r in range(1000)])
    s_hi = mean([co.segregating_sites(8, 10.0, seed=r + 1)[0] for r in range(1000)])
    check(f"larger theta -> more segregating sites ({s_lo:.1f} < {s_hi:.1f})", s_lo < s_hi)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

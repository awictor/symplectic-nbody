"""Validate Ehrenfest urn: binomial stationary, detailed balance, relaxation to N/2, entropy rise, recurrence."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import ehrenfest_urn as eu


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Ehrenfest urn tests")

    # --- long-run occupation matches Binomial(N, 1/2) ---
    n = 10
    counts = eu.simulate(n, n, 200000, seed=1)   # start all in A
    hist = eu.occupation_histogram(counts[1000:], n)  # drop burn-in
    total = sum(hist)
    emp = [h / total for h in hist]
    pi = eu.stationary_distribution(n)
    max_err = max(abs(emp[k] - pi[k]) for k in range(n + 1))
    check(f"occupation ~ Binomial(N,1/2) (max err {max_err:.4f})", max_err < 0.02)

    # --- stationary distribution sums to 1 and peaks at N/2 ---
    check("stationary sums to 1", abs(sum(pi) - 1.0) < 1e-12)
    check("stationary peaks at N/2", pi.index(max(pi)) == n // 2)

    # --- detailed balance holds (reversible chain) ---
    check("detailed balance satisfied", eu.detailed_balance_residual(20) < 1e-15)

    # --- relaxation: starting all-in-one, mean count moves toward N/2 ---
    n = 20
    # average many trajectories to get E[k_t]
    T = 40
    n_runs = 500
    mean_k = [0.0] * (T + 1)
    for r in range(n_runs):
        c = eu.simulate(n, n, T, seed=r + 1)
        for t in range(T + 1):
            mean_k[t] += c[t]
    mean_k = [m / n_runs for m in mean_k]
    check("starts at N", abs(mean_k[0] - n) < 1e-9)
    check("relaxes toward N/2", abs(mean_k[-1] - n / 2) < 2.0)
    check("relaxation is monotone (decreasing)", all(mean_k[t] >= mean_k[t + 1] - 0.5 for t in range(T)))

    # --- expected next count matches the analytic drift ---
    # simulate one step many times from a fixed k, compare mean to expected_next
    n = 10
    k = 8
    nexts = []
    for r in range(5000):
        c = eu.simulate(n, k, 1, seed=r + 1)
        nexts.append(c[1])
    emp_next = sum(nexts) / len(nexts)
    check(f"expected next ~ analytic ({emp_next:.3f} vs {eu.expected_next(n, k):.3f})",
          abs(emp_next - eu.expected_next(n, k)) < 0.1)
    # analytic drift moves toward n/2
    check("drift toward N/2 from above", eu.expected_next(10, 8) < 8)
    check("drift toward N/2 from below", eu.expected_next(10, 2) > 2)
    check("N/2 is the drift fixed point", abs(eu.expected_next(10, 5) - 5) < 1e-12)

    # --- entropy rises as the system spreads from all-in-one ---
    # track the distribution over an ensemble at t=0 (all in A) vs later
    n = 12
    T = 30
    n_runs = 2000
    dist0 = [0] * (n + 1)
    distT = [0] * (n + 1)
    for r in range(n_runs):
        c = eu.simulate(n, n, T, seed=r + 1)
        dist0[c[0]] += 1
        distT[c[-1]] += 1
    h0 = eu.entropy(dist0)
    hT = eu.entropy(distT)
    check(f"entropy rises from all-in-one ({h0:.2f} -> {hT:.2f})", hT > h0 + 1.0)

    # --- mean recurrence time to all-in-one is 2^N ---
    check("mean recurrence time == 2^N", eu.mean_recurrence_time(10) == 2 ** 10)
    # and equals 1/pi_0 (pi_0 = 1/2^N)
    pi10 = eu.stationary_distribution(10)
    check("recurrence time == 1/pi_0", abs(eu.mean_recurrence_time(10) - 1 / pi10[0]) < 1e-6)

    # --- transition probabilities ---
    check("down prob k/n", abs(eu.transition_prob(10, 7, 6) - 0.7) < 1e-12)
    check("up prob (n-k)/n", abs(eu.transition_prob(10, 7, 8) - 0.3) < 1e-12)
    check("no jump of size 2", eu.transition_prob(10, 5, 7) == 0.0)

    # --- deterministic ---
    check("deterministic", eu.simulate(10, 10, 100, seed=42) == eu.simulate(10, 10, 100, seed=42))

    # --- counts stay in [0, n] ---
    c = eu.simulate(15, 0, 5000, seed=3)
    check("counts within [0, n]", all(0 <= k <= 15 for k in c))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

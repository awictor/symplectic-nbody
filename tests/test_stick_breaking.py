"""Validate stick-breaking: weights sum to 1, geometric expected weights, concentration effect, label sampling."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import stick_breaking as sb


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def mean(xs):
    return sum(xs) / len(xs)


def main():
    print("Stick-breaking tests")

    # --- weights are positive and partial sums approach 1 ---
    w = sb.weights(1.0, 200, seed=1)
    check("weights positive", all(p > 0 for p in w))
    check("weights decreasing on average (mass front-loaded)", sum(w[:10]) > sum(w[100:110]))
    check(f"partial sum approaches 1 ({sum(w):.4f})", sum(w) > 0.99)

    # --- empirical mean weight matches the geometric formula ---
    alpha = 2.0
    n_runs = 3000
    K = 5
    acc = [0.0] * K
    for r in range(n_runs):
        w = sb.weights(alpha, K, seed=(r + 1) * 7919)
        for k in range(K):
            acc[k] += w[k]
    for k in range(K):
        emp = acc[k] / n_runs
        theo = sb.expected_weight(alpha, k + 1)
        check(f"E[pi_{k+1}] ~ geometric ({emp:.4f} vs {theo:.4f})", abs(emp - theo) < 0.01)

    # --- expected weights sum toward 1 as k grows ---
    total_exp = sum(sb.expected_weight(alpha, k) for k in range(1, 200))
    check(f"expected weights sum -> 1 ({total_exp:.4f})", total_exp > 0.99)

    # --- expected residual after k breaks ---
    check("residual formula", abs(sb.expected_residual(2.0, 3) - (2 / 3) ** 3) < 1e-12)
    # residual + captured expectation = 1
    cap = sum(sb.expected_weight(2.0, k) for k in range(1, 6))
    check("captured + residual == 1", abs(cap + sb.expected_residual(2.0, 5) - 1.0) < 1e-9)

    # --- concentration: small alpha needs fewer weights for 95% mass ---
    k_small = mean([sb.weights_for_mass(0.5, 0.95, seed=(r + 1) * 7919) for r in range(500)])
    k_large = mean([sb.weights_for_mass(5.0, 0.95, seed=(r + 1) * 7919) for r in range(500)])
    check(f"small alpha -> fewer weights for 95% ({k_small:.1f} < {k_large:.1f})", k_small < k_large)

    # --- sampling labels reproduces the weights as empirical frequencies ---
    alpha = 1.5
    w = sb.weights(alpha, 50, seed=3)
    labels = sb.sample_labels(w, 20000, seed=5)
    total = sum(w)
    from collections import Counter
    cnt = Counter(labels)
    # compare the top few weights to their empirical frequency
    ok = True
    for k in range(5):
        emp_freq = cnt[k] / len(labels)
        target = w[k] / total
        if abs(emp_freq - target) > 0.02:
            ok = False
    check("label frequencies match weights", ok)

    # --- Beta(1,alpha) sampler: mean is 1/(1+alpha) ---
    rng = sb._Rng(1)
    betas = [rng.beta_1_alpha(3.0) for _ in range(20000)]
    check(f"Beta(1,alpha) mean ~ 1/(1+alpha) ({mean(betas):.3f} vs {1/4:.3f})",
          abs(mean(betas) - 0.25) < 0.02)

    # --- smaller alpha -> first weight larger on average (bigger early pieces) ---
    w1_small = mean([sb.weights(0.5, 1, seed=(r + 1) * 7919)[0] for r in range(2000)])
    w1_large = mean([sb.weights(5.0, 1, seed=(r + 1) * 7919)[0] for r in range(2000)])
    check(f"small alpha -> bigger first weight ({w1_small:.3f} > {w1_large:.3f})", w1_small > w1_large)

    # --- deterministic ---
    check("deterministic", sb.weights(2.0, 20, seed=42) == sb.weights(2.0, 20, seed=42))

    # --- first expected weight is 1/(1+alpha) ---
    check("E[pi_1] == 1/(1+alpha)", abs(sb.expected_weight(3.0, 1) - 0.25) < 1e-12)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

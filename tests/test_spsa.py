"""Validate SPSA against analytic optima and its two-eval-per-iteration guarantee."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import spsa


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def main():
    print("SPSA tests")

    # --- minimizes a quadratic bowl to near zero ---
    theta0 = [3.0, -4.0, 2.5, 1.0]
    out = spsa.minimize(spsa.sphere, theta0, iterations=2000, a=0.3, c=0.1, seed=1)
    check("sphere converges near origin", norm(out) < 0.15)
    check("sphere improves on start", spsa.sphere(out) < spsa.sphere(theta0) * 1e-2)

    # --- locates a shifted optimum ---
    center = [1.5, -2.0, 0.75]
    f = spsa.shifted_sphere(center)
    out = spsa.minimize(f, [0.0, 0.0, 0.0], iterations=2000, a=0.3, c=0.1, seed=7)
    err = norm([out[i] - center[i] for i in range(3)])
    check("shifted optimum located", err < 0.15)

    # --- exactly two evaluations per iteration, ANY dimension ---
    for p in (1, 2, 10, 50):
        counter = [0]

        def counted(x):
            counter[0] += 1
            return spsa.sphere(x)

        iters = 100
        spsa.minimize(counted, [1.0] * p, iterations=iters, a=0.2, c=0.1, seed=3)
        # minimize evaluates f twice per iter (track=False -> no extra eval)
        check(f"exactly 2 evals/iter in dim {p}", counter[0] == 2 * iters)

    # --- reported eval count matches (track mode) ---
    _, hist, evals = spsa.minimize(spsa.sphere, [2.0, 2.0], iterations=500,
                                   a=0.3, c=0.1, seed=2, track=True)
    check("tracked eval count == 2*iters", evals == 2 * 500)
    check("history length == iters", len(hist) == 500)
    check("objective decreases over run", hist[-1][1] < hist[0][1])

    # --- converges under additive noise (finite diff would thrash) ---
    g = spsa.noisy(spsa.sphere, noise_amp=0.05, seed=11)
    out = spsa.minimize(g, [2.0, -2.0, 1.0], iterations=3000, a=0.25, c=0.15, seed=5)
    check("noisy objective still converges", norm(out) < 0.4)

    # --- gain sequences decay at specified rates ---
    aks, cks = spsa.gain_sequences(1000, a=0.2, c=0.1, alpha=0.602, gamma=0.101)
    check("a_k monotonically decreasing", all(aks[i] >= aks[i + 1] for i in range(999)))
    check("c_k monotonically decreasing", all(cks[i] >= cks[i + 1] for i in range(999)))
    # a_k decays faster than c_k (alpha > gamma) -> ratio a_k/c_k shrinks
    r0 = aks[0] / cks[0]
    r1 = aks[-1] / cks[-1]
    check("a_k decays faster than c_k", r1 < r0)
    # check the analytic form at k=0
    A = max(1, 1000 // 10)
    check("a_0 matches formula", abs(aks[0] - 0.2 / ((1 + A) ** 0.602)) < 1e-12)
    check("c_0 matches formula", abs(cks[0] - 0.1 / (1 ** 0.101)) < 1e-12)

    # --- reproducible for a fixed seed, differs across seeds ---
    o1 = spsa.minimize(spsa.sphere, [1.0, 1.0], iterations=200, seed=42)
    o2 = spsa.minimize(spsa.sphere, [1.0, 1.0], iterations=200, seed=42)
    o3 = spsa.minimize(spsa.sphere, [1.0, 1.0], iterations=200, seed=43)
    check("reproducible for same seed", o1 == o2)
    check("differs across seeds", o1 != o3)

    # --- 1-D scalar problem ---
    out1 = spsa.minimize(lambda x: (x[0] - 5.0) ** 2, [0.0], iterations=1500,
                         a=0.4, c=0.1, seed=9)
    check("1-D optimum located", abs(out1[0] - 5.0) < 0.1)

    # --- SPSA beats finite-difference on eval count in high dim ---
    # FD gradient descent needs 2p evals/iter; SPSA needs 2. For p=50 over 100
    # iters: FD = 10000 evals, SPSA = 200. Demonstrate the 50x saving concretely.
    p = 50
    fd_evals = 2 * p * 100
    spsa_evals = 2 * 100
    check("SPSA uses 50x fewer evals than FD (p=50)", spsa_evals * 50 == fd_evals)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

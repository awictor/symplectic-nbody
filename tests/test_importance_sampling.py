"""Tests for importance sampling: rare-tail accuracy vs naive MC, variance reduction, ESS, self-normalized."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import importance_sampling as IS  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    # ---- 1. rare tail probability: IS is accurate where naive MC fails ------------------
    t = 4.0
    exact = IS.normal_tail_exact(t)
    ise, istd, ess = IS.is_tail_probability(t, n=20000, seed=1)
    naive = IS.naive_tail_probability(t, n=20000, seed=1)
    check("IS recovers P(Z>4) within 5%", abs(ise - exact) / exact < 0.05,
          f"is {ise:.3e} exact {exact:.3e}")
    check("naive MC misses the rare tail (0 or huge relative error)",
          naive == 0.0 or abs(naive - exact) / exact > abs(ise - exact) / exact,
          f"naive {naive:.3e}")

    # ---- 2. IS standard error is small for the rare event -------------------------------
    check("IS standard error is tiny", istd < exact, f"stderr {istd:.2e} vs p {exact:.3e}")

    # ---- 3. even deeper tail P(Z>5): still accurate -------------------------------------
    exact5 = IS.normal_tail_exact(5.0)
    ise5, _, _ = IS.is_tail_probability(5.0, n=30000, seed=7)
    check("IS recovers P(Z>5) within 10%", abs(ise5 - exact5) / exact5 < 0.10,
          f"is {ise5:.3e} exact {exact5:.3e}")

    # ---- 4. ordinary IS of an expectation matches the analytic value --------------------
    # E[X^2] for N(0,1) = 1, sampling from a wider proposal N(0,2)
    f = lambda x: x * x
    p = lambda x: IS._normal_pdf(x, 0, 1)
    q = lambda x: IS._normal_pdf(x, 0, 2)
    est, se, ess = IS.importance_estimate(f, p, q, lambda rng: rng.normal(0, 2), n=30000, seed=2)
    check("IS estimate of E[X^2] ~ 1", abs(est - 1.0) < 0.05, f"{est:.3f}")

    # ---- 5. q == p gives unit weights: ESS == n, zero-variance weight set ---------------
    est2, se2, ess2 = IS.importance_estimate(
        lambda x: 1.0, p, p, lambda rng: rng.normal(0, 1), n=2000, seed=3)
    check("q==p -> ESS == n", abs(ess2 - 2000) < 1e-6, f"{ess2}")
    check("q==p -> estimate of constant is exact", abs(est2 - 1.0) < 1e-9, f"{est2}")

    # ---- 6. effective sample size formula ----------------------------------------------
    check("ESS of equal weights == n", abs(IS.effective_sample_size([1.0] * 50) - 50) < 1e-9)
    # one dominant weight -> ESS near 1
    w = [1000.0] + [0.001] * 99
    check("ESS of one dominant weight ~ 1", IS.effective_sample_size(w) < 1.1,
          f"{IS.effective_sample_size(w):.3f}")

    # ---- 7. ESS degrades as the proposal mismatches the target --------------------------
    # for the tail estimator, shifting the proposal far past t hurts ESS
    _, _, ess_good = IS.is_tail_probability(4.0, n=10000, seed=4, shift=4.0)
    _, _, ess_bad = IS.is_tail_probability(4.0, n=10000, seed=4, shift=10.0)
    check("ESS is higher for a well-matched proposal", ess_good > ess_bad,
          f"good {ess_good:.1f} bad {ess_bad:.1f}")

    # ---- 8. self-normalized IS recovers a posterior mean --------------------------------
    # target (unnormalized): N(2, 1) times a constant; proposal N(0, 3); estimate E[X] = 2
    p_un = lambda x: 3.7 * IS._normal_pdf(x, 2.0, 1.0)   # arbitrary unknown constant
    q2 = lambda x: IS._normal_pdf(x, 0.0, 3.0)
    est_sn, ess_sn = IS.self_normalized_estimate(
        lambda x: x, p_un, q2, lambda rng: rng.normal(0, 3), n=30000, seed=5)
    check("self-normalized IS recovers the mean (2.0)", abs(est_sn - 2.0) < 0.08, f"{est_sn:.3f}")

    # ---- 9. IS is unbiased: averaging over seeds converges to the truth -----------------
    ests = [IS.is_tail_probability(3.5, n=5000, seed=s)[0] for s in range(1, 21)]
    avg = sum(ests) / len(ests)
    check("IS is unbiased (seed-average ~ exact)", abs(avg - IS.normal_tail_exact(3.5)) < 0.06 * IS.normal_tail_exact(3.5),
          f"avg {avg:.3e} exact {IS.normal_tail_exact(3.5):.3e}")

    # ---- 10. variance reduction: IS variance << naive variance for the rare event -------
    # use t=3 so naive MC actually sees some tail samples (its variance is meaningful);
    # IS relative to its own estimate should still be dramatically tighter.
    t10 = 3.0
    exact10 = IS.normal_tail_exact(t10)
    is_ests = [IS.is_tail_probability(t10, n=3000, seed=s)[0] for s in range(1, 41)]
    naive_ests = [IS.naive_tail_probability(t10, n=3000, seed=s) for s in range(1, 41)]

    def var(a):
        m = sum(a) / len(a)
        return sum((x - m) ** 2 for x in a) / len(a)
    check("IS variance far below naive MC variance", var(is_ests) < var(naive_ests) * 0.2,
          f"is {var(is_ests):.2e} naive {var(naive_ests):.2e}")
    check("IS seed-average matches exact at t=3",
          abs(sum(is_ests) / len(is_ests) - exact10) < 0.05 * exact10)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

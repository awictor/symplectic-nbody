"""Tests for HMC: recovers Gaussian moments, correlation, high acceptance, beats random walk."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hmc import (  # noqa: E402
    hmc,
    random_walk_metropolis,
    sample_mean,
    sample_cov,
    autocorrelation,
    finite_diff_grad,
)


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
    # ---- 1. standard normal: mean ~ 0, variance ~ 1 -------------------------------------
    def logp_std(q):
        return -0.5 * sum(x * x for x in q)

    def grad_std(q):
        return [-x for x in q]

    samples, rate = hmc(logp_std, [0.0], 3000, step_size=0.3, n_leapfrog=15,
                        grad=grad_std, seed=7, burn_in=500)
    m = sample_mean(samples)[0]
    v = sample_cov(samples)[0][0]
    check("std normal: mean ~ 0", abs(m) < 0.1, f"{m:.4f}")
    check("std normal: variance ~ 1", abs(v - 1) < 0.15, f"{v:.4f}")
    check("std normal: acceptance high", rate > 0.8, f"{rate:.3f}")

    # ---- 2. correlated 2-D Gaussian -----------------------------------------------------
    # covariance [[1, 0.8],[0.8, 1]], precision = inv(cov)
    rho = 0.8
    det = 1 - rho * rho
    # precision matrix
    P = [[1 / det, -rho / det], [-rho / det, 1 / det]]

    def logp_corr(q):
        x, y = q
        return -0.5 * (P[0][0] * x * x + 2 * P[0][1] * x * y + P[1][1] * y * y)

    def grad_corr(q):
        x, y = q
        return [-(P[0][0] * x + P[0][1] * y), -(P[1][0] * x + P[1][1] * y)]

    samples, rate = hmc(logp_corr, [0.0, 0.0], 4000, step_size=0.25, n_leapfrog=20,
                        grad=grad_corr, seed=11, burn_in=1000)
    mean = sample_mean(samples)
    cov = sample_cov(samples)
    check("correlated: means ~ 0", abs(mean[0]) < 0.15 and abs(mean[1]) < 0.15, f"{mean}")
    check("correlated: variances ~ 1", abs(cov[0][0] - 1) < 0.2 and abs(cov[1][1] - 1) < 0.2,
          f"{cov[0][0]:.3f}, {cov[1][1]:.3f}")
    recovered_rho = cov[0][1] / math.sqrt(cov[0][0] * cov[1][1])
    check("correlated: correlation ~ 0.8", abs(recovered_rho - rho) < 0.1, f"{recovered_rho:.3f}")

    # ---- 3. HMC mixes better than random-walk Metropolis on the correlated target -------
    hmc_s, _ = hmc(logp_corr, [0.0, 0.0], 2000, step_size=0.25, n_leapfrog=20,
                   grad=grad_corr, seed=3, burn_in=500)
    rw_s, _ = random_walk_metropolis(logp_corr, [0.0, 0.0], 2000, step_size=0.4, seed=3, burn_in=500)
    ac_hmc = abs(autocorrelation(hmc_s, 0, 5))
    ac_rw = abs(autocorrelation(rw_s, 0, 5))
    check("HMC lower autocorrelation than random walk", ac_hmc < ac_rw,
          f"HMC {ac_hmc:.3f} vs RW {ac_rw:.3f}")

    # ---- 4. smaller step size -> higher acceptance --------------------------------------
    _, rate_big = hmc(logp_std, [0.0], 1000, step_size=0.8, n_leapfrog=10, grad=grad_std, seed=5)
    _, rate_small = hmc(logp_std, [0.0], 1000, step_size=0.1, n_leapfrog=10, grad=grad_std, seed=5)
    check("smaller step size raises acceptance", rate_small >= rate_big, f"{rate_big:.3f} -> {rate_small:.3f}")

    # ---- 5. finite-difference gradient matches analytic ---------------------------------
    q = [1.5, -0.7]
    fd = finite_diff_grad(logp_corr, q)
    an = grad_corr(q)
    check("finite-diff gradient matches analytic", all(abs(fd[i] - an[i]) < 1e-4 for i in range(2)))

    # ---- 6. HMC works with finite-difference gradient (no analytic grad) ----------------
    samples, rate = hmc(logp_std, [0.0], 1500, step_size=0.3, n_leapfrog=12, seed=9, burn_in=300)
    check("HMC with finite-diff gradient recovers std normal",
          abs(sample_mean(samples)[0]) < 0.15 and abs(sample_cov(samples)[0][0] - 1) < 0.2,
          f"mean {sample_mean(samples)[0]:.3f}, var {sample_cov(samples)[0][0]:.3f}")

    # ---- 7. a non-Gaussian target: recover mean of an off-center normal -----------------
    def logp_shift(q):
        return -0.5 * ((q[0] - 3.0) ** 2) / 4.0  # N(3, 2^2)
    samples, _ = hmc(logp_shift, [0.0], 5000, step_size=0.6, n_leapfrog=25, seed=13, burn_in=1000)
    m = sample_mean(samples)[0]
    v = sample_cov(samples)[0][0]
    check("off-center normal mean ~ 3", abs(m - 3) < 0.3, f"{m:.3f}")
    check("off-center normal variance ~ 4", abs(v - 4) < 1.0, f"{v:.3f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

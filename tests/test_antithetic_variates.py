"""Tests for antithetic variates: unbiasedness, monotone reduction, symmetric no-help, Gaussian reflection."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import antithetic_variates as AV  # noqa: E402


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
    # ---- 1. monotone integrand e^U: unbiased and variance reduced -----------------------
    f = lambda u: math.exp(u)
    true = math.e - 1
    est, se, red = AV.estimate_uniform(f, n_pairs=15000, seed=1)
    check("antithetic estimate of E[e^U] ~ e-1", abs(est - true) < 0.005, f"{est:.5f}")
    check("variance reduced for monotone f (<1)", red < 1.0, f"{red:.4f}")
    check("reduction is large for e^U (<0.1)", red < 0.1, f"{red:.4f}")

    # ---- 2. pair covariance is negative for a monotone integrand ------------------------
    cov = AV.pair_covariance(f, 15000, 1)
    check("pair covariance negative for monotone f", cov < 0, f"{cov:.4f}")

    # ---- 3. a symmetric integrand gets no benefit (reduction >= ~1) ---------------------
    g = lambda u: (u - 0.5) ** 2                  # symmetric about 0.5: f(u)=f(1-u)
    _, _, red_g = AV.estimate_uniform(g, 15000, seed=2)
    check("symmetric integrand: no variance reduction", red_g > 0.9, f"{red_g:.4f}")
    cov_g = AV.pair_covariance(g, 15000, 2)
    check("symmetric integrand: non-negative pair covariance", cov_g > -1e-3, f"{cov_g:.5f}")

    # ---- 4. linear integrand: perfect cancellation, near-zero variance ------------------
    lin = lambda u: 3 * u + 1                     # E = 3*0.5 + 1 = 2.5
    est_l, se_l, red_l = AV.estimate_uniform(lin, 5000, seed=3)
    check("linear integrand estimate exact", abs(est_l - 2.5) < 1e-9, f"{est_l}")
    check("linear integrand variance ~ 0 (perfect antithetic)", se_l < 1e-9, f"{se_l:.2e}")

    # ---- 5. antithetic beats plain MC at equal evaluations for a monotone integrand -----
    _, se_plain = AV.plain_estimate(f, n=30000, seed=5)
    _, se_anti, _ = AV.estimate_uniform(f, n_pairs=15000, seed=5)  # 30000 evals
    check("antithetic stderr below plain MC (equal evals)", se_anti < se_plain,
          f"anti {se_anti:.5f} plain {se_plain:.5f}")

    # ---- 6. Gaussian reflection recovers E[f(Z)] ----------------------------------------
    h = lambda z: math.exp(z)                     # E[e^Z] = e^{1/2}
    est_h, se_h, red_h = AV.estimate_normal(h, 40000, seed=6)
    check("E[e^Z] ~ e^{1/2}", abs(est_h - math.exp(0.5)) < 0.02, f"{est_h:.5f}")
    check("Gaussian antithetic reduces variance for monotone f", red_h < 1.0, f"{red_h:.4f}")

    # ---- 7. an odd function of Z: antithetic gives near-perfect cancellation ------------
    odd = lambda z: z ** 3 + 2 * z                # odd -> E = 0, f(z)+f(-z) = 0 exactly
    est_o, se_o, _ = AV.estimate_normal(odd, 5000, seed=7)
    check("odd function estimate ~ 0", abs(est_o) < 1e-9, f"{est_o}")
    check("odd function variance ~ 0 (perfect reflection)", se_o < 1e-9, f"{se_o:.2e}")

    # ---- 8. multidimensional monotone integrand ----------------------------------------
    fd = lambda u: math.exp(sum(u))               # separable, monotone in each coordinate
    est_d, se_d, red_d = AV.estimate_uniform(fd, 15000, dim=3, seed=8)
    true_d = (math.e - 1) ** 3
    check("3-D E[e^{sum U}] ~ (e-1)^3", abs(est_d - true_d) < 0.05, f"{est_d:.4f} vs {true_d:.4f}")
    check("3-D monotone: variance reduced", red_d < 1.0, f"{red_d:.4f}")

    # ---- 9. reproducibility -------------------------------------------------------------
    a = AV.estimate_uniform(f, 1000, seed=42)
    b = AV.estimate_uniform(f, 1000, seed=42)
    check("same seed -> identical estimate", abs(a[0] - b[0]) < 1e-15)
    c = AV.estimate_uniform(f, 1000, seed=43)
    check("different seed -> different estimate", abs(a[0] - c[0]) > 1e-12)

    # ---- 10. reduction factor tracks the pair-covariance formula ------------------------
    # reduction = (Var(f) + Cov(f(U),f(1-U))) / (2 * Var(f)) approximately; check it is < 0.5
    # for a strongly monotone integrand (cov close to -Var)
    check("strong monotone reduction below 0.5", red < 0.5, f"{red:.4f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

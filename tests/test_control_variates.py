"""Tests for control variates: unbiasedness, 1-rho^2 reduction, optimal coefficient, multi-CV, no-corr case."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import control_variates as CV  # noqa: E402


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
    rng = CV._Rng(1)
    n = 30000
    U = [rng.uniform() for _ in range(n)]
    f = [math.exp(u) for u in U]                 # E[e^U] = e - 1
    g = U                                        # control with known mean 0.5
    true = math.e - 1

    # ---- 1. unbiasedness: CV estimate matches the truth ---------------------------------
    est, se, red = CV.estimate(f, g, 0.5)
    check("CV estimate unbiased (~ e-1)", abs(est - true) < 0.005, f"{est:.5f} vs {true:.5f}")
    plain = CV._mean(f)
    check("plain MC also unbiased", abs(plain - true) < 0.01, f"{plain:.5f}")

    # ---- 2. empirical variance reduction matches theory 1 - rho^2 -----------------------
    theory = CV.theoretical_reduction(f, g)
    check("empirical reduction == 1 - rho^2", abs(red - theory) < 1e-6, f"emp {red:.5f} thy {theory:.5f}")

    # ---- 3. strong correlation -> large reduction ---------------------------------------
    rho = CV.correlation(f, g)
    check("e^U and U are highly correlated", rho > 0.98, f"{rho:.4f}")
    check("reduction is substantial (>90%)", red < 0.1, f"{red:.4f}")

    # ---- 4. optimal coefficient equals Cov/Var ------------------------------------------
    c_star = CV.optimal_coefficient(f, g)
    cov = CV._cov(f, g)
    var = CV._var(g)
    check("c* == Cov(f,g)/Var(g)", abs(c_star - cov / var) < 1e-12, f"{c_star}")
    # c* minimizes variance: perturbing c raises the controlled variance
    _, _, red_opt = CV.estimate(f, g, 0.5, c=c_star)
    _, _, red_off1 = CV.estimate(f, g, 0.5, c=c_star + 0.3)
    _, _, red_off2 = CV.estimate(f, g, 0.5, c=c_star - 0.3)
    check("c* minimizes controlled variance", red_opt <= red_off1 and red_opt <= red_off2,
          f"opt {red_opt:.4f} +{red_off1:.4f} -{red_off2:.4f}")

    # ---- 5. an uncorrelated control gives ~no reduction ---------------------------------
    rng2 = CV._Rng(999)
    h = [rng2.uniform() for _ in range(n)]       # independent of f
    _, _, red_h = CV.estimate(f, h, 0.5)
    check("uncorrelated control -> reduction ~ 1", 0.9 < red_h <= 1.05, f"{red_h:.4f}")

    # ---- 6. still unbiased with the uncorrelated control --------------------------------
    est_h, _, _ = CV.estimate(f, h, 0.5)
    check("unbiased even with a useless control", abs(est_h - true) < 0.01, f"{est_h:.5f}")

    # ---- 7. multiple control variates reduce at least as much as the best single --------
    g2 = [u * u for u in U]                       # E[U^2] = 1/3
    est_m, se_m, red_m, coeffs = CV.estimate_multi(f, [U, g2], [0.5, 1.0 / 3.0])
    check("multi-CV unbiased", abs(est_m - true) < 0.005, f"{est_m:.5f}")
    _, _, red_single = CV.estimate(f, g, 0.5)
    check("multi-CV reduces >= best single", red_m <= red_single + 1e-9,
          f"multi {red_m:.5f} single {red_single:.5f}")
    check("multi-CV returns one coefficient per control", len(coeffs) == 2)

    # ---- 8. a second problem: E[X^2] for X ~ N(0,1) = 1, control g = X (mean 0) ----------
    rng3 = CV._Rng(7)
    X = [rng3.normal() for _ in range(n)]
    fx = [x * x for x in X]
    est_x, _, red_x = CV.estimate(fx, X, 0.0)     # g=X has mean 0, but is uncorrelated with X^2
    check("E[X^2] ~ 1", abs(est_x - 1.0) < 0.03, f"{est_x:.4f}")
    # X and X^2 are uncorrelated for a symmetric distribution -> little reduction
    check("symmetric control gives little reduction", red_x > 0.9, f"{red_x:.4f}")
    # but g = X^2-shifted-partner: use |X| which correlates with X^2
    absX = [abs(x) for x in X]
    mean_absX = math.sqrt(2 / math.pi)            # E[|Z|] for standard normal
    _, _, red_abs = CV.estimate(fx, absX, mean_absX)
    check("|X| is a useful control for X^2", red_abs < red_x, f"abs {red_abs:.4f} vs {red_x:.4f}")

    # ---- 9. reduction factor is in (0, 1] -----------------------------------------------
    check("reduction factor in (0,1]", 0 < red <= 1.0 + 1e-9)

    # ---- 10. standard error shrinks with the reduction ----------------------------------
    _, se_plain, _ = CV.estimate(f, [0.0] * n, 0.0)   # zero control = plain MC
    check("CV standard error below plain MC", se < se_plain, f"cv {se:.5f} plain {se_plain:.5f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

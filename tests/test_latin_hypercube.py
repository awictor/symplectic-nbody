"""Tests for LHS: one point per stratum, in unit cube, lower variance than MC, maximin, centered."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from latin_hypercube import (  # noqa: E402
    latin_hypercube,
    maximin_lhs,
    min_pairwise_distance,
    integrate,
    stratification_ok,
    monte_carlo_integrate,
    estimator_variance,
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
    # ---- 1. every axis has exactly one point per stratum (the defining property) --------
    ok = True
    for seed in range(30):
        for n, dim in [(10, 3), (20, 5), (7, 2), (50, 4)]:
            design = latin_hypercube(n, dim, seed=seed)
            if not stratification_ok(design):
                ok = False
    check("every axis exactly one point per stratum", ok)

    # ---- 2. samples lie in the unit cube ------------------------------------------------
    design = latin_hypercube(30, 4, seed=7)
    check("all samples in [0,1]^d",
          all(0 <= design[i][j] <= 1 for i in range(30) for j in range(4)))

    # ---- 3. LHS integration has lower variance (MSE) than plain Monte Carlo -------------
    # an additive function where LHS shines: f = sum sin(pi x_i)
    def f_add(x):
        return sum(math.sin(math.pi * xi) for xi in x)
    dim = 4
    # true integral over unit cube: dim * integral_0^1 sin(pi x) dx = dim * 2/pi
    true = dim * 2 / math.pi
    n = 40
    mse_lhs = estimator_variance(integrate, f_add, dim, n, true, trials=60)
    mse_mc = estimator_variance(monte_carlo_integrate, f_add, dim, n, true, trials=60)
    check("LHS MSE < Monte Carlo MSE on additive function",
          mse_lhs < mse_mc, f"LHS {mse_lhs:.4e} vs MC {mse_mc:.4e}")
    check("LHS at least 3x better here", mse_lhs * 3 < mse_mc, f"ratio {mse_mc/mse_lhs:.1f}")

    # ---- 4. both integrators are approximately unbiased ---------------------------------
    est_lhs = sum(integrate(f_add, dim, 100, seed=1000 + t) for t in range(20)) / 20
    check("LHS estimate unbiased (near true)", abs(est_lhs - true) < 0.05, f"{est_lhs:.3f} vs {true:.3f}")

    # ---- 5. integrate a simple known function -------------------------------------------
    # f = product of x_i over unit cube = (1/2)^dim
    def f_prod(x):
        p = 1.0
        for xi in x:
            p *= xi
        return p
    est = integrate(f_prod, 3, 2000, seed=3)
    check("LHS integrates product of x_i to (1/2)^3", abs(est - 0.125) < 0.01, f"{est:.4f}")

    # ---- 6. maximin has larger min distance than a plain draw ---------------------------
    plain = latin_hypercube(15, 3, seed=42)
    mm, mm_d = maximin_lhs(15, 3, seed=42, candidates=30)
    check("maximin min-distance >= plain LHS min-distance",
          mm_d >= min_pairwise_distance(plain), f"maximin {mm_d:.4f} vs plain {min_pairwise_distance(plain):.4f}")
    check("maximin design is still a valid LHS", stratification_ok(mm))

    # ---- 7. centered LHS points sit at stratum midpoints --------------------------------
    design = latin_hypercube(10, 2, seed=5, centered=True)
    # each coordinate should be (k + 0.5)/10 for some integer k
    ok = True
    for i in range(10):
        for j in range(2):
            v = design[i][j] * 10 - 0.5
            if abs(v - round(v)) > 1e-9:
                ok = False
    check("centered LHS points at stratum midpoints", ok)

    # ---- 8. determinism + edge cases ----------------------------------------------------
    check("deterministic under seed", latin_hypercube(10, 3, seed=1) == latin_hypercube(10, 3, seed=1))
    check("different seeds differ", latin_hypercube(10, 3, seed=1) != latin_hypercube(10, 3, seed=2))
    check("n=1 single sample", len(latin_hypercube(1, 3, seed=1)) == 1)
    try:
        latin_hypercube(0, 3)
        check("n=0 raises", False)
    except ValueError:
        check("n=0 raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

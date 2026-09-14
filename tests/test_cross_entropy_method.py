"""Tests for the cross-entropy method: benchmark convergence, variance collapse, elite behavior."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cross_entropy_method as CEM  # noqa: E402


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
    # ---- 1. sphere / quadratic bowl -> origin -------------------------------------------
    x = CEM.optimize(CEM.sphere, [5.0, 5.0, 5.0], std=[3] * 3, population=60, iterations=80, seed=1)
    check("sphere minimized to ~0", CEM.sphere(x) < 1e-4, f"{CEM.sphere(x):.2e}")
    check("sphere minimizer near origin", all(abs(v) < 0.02 for v in x), f"{[round(v,3) for v in x]}")

    # ---- 2. Rastrigin (multimodal trap) -> origin ---------------------------------------
    x = CEM.optimize(CEM.rastrigin, [3.0, -2.0], std=[4, 4], population=120, iterations=200, seed=3)
    check("Rastrigin escapes local minima to ~0", CEM.rastrigin(x) < 1.0, f"{CEM.rastrigin(x):.4f}")

    # ---- 3. Rosenbrock banana approaches (1, 1) -----------------------------------------
    x = CEM.optimize(CEM.rosenbrock, [-1.0, 1.0], std=[2, 2], population=200, iterations=400,
                     smoothing=0.5, seed=2)
    check("Rosenbrock approaches its minimum", CEM.rosenbrock(x) < 0.1, f"{CEM.rosenbrock(x):.4f}")
    check("Rosenbrock minimizer near (1,1)", abs(x[0] - 1) < 0.3 and abs(x[1] - 1) < 0.3,
          f"{[round(v,3) for v in x]}")

    # ---- 4. the sampling variance collapses toward the optimum --------------------------
    _, hist = CEM.optimize(CEM.sphere, [5.0, 5.0], std=[3, 3], population=50, iterations=60,
                           seed=4, track=True)
    check("variance shrinks over iterations", hist[-1][1] < hist[0][1] * 1e-3,
          f"start {hist[0][1]:.3f} end {hist[-1][1]:.2e}")
    check("best value improves monotonically (non-increasing)",
          all(hist[i][0] >= hist[i + 1][0] - 1e-12 for i in range(len(hist) - 1)))

    # ---- 5. a custom objective: minimize (x-3)^2 + (y+2)^2 ------------------------------
    f = lambda p: (p[0] - 3) ** 2 + (p[1] + 2) ** 2
    x = CEM.optimize(f, [0.0, 0.0], std=[5, 5], population=50, iterations=100, seed=5)
    check("custom quadratic finds (3, -2)", abs(x[0] - 3) < 0.02 and abs(x[1] + 2) < 0.02,
          f"{[round(v,3) for v in x]}")

    # ---- 6. reproducibility -------------------------------------------------------------
    a = CEM.optimize(CEM.sphere, [4.0, 4.0], population=40, iterations=30, seed=42)
    b = CEM.optimize(CEM.sphere, [4.0, 4.0], population=40, iterations=30, seed=42)
    check("same seed -> identical result", a == b)
    c = CEM.optimize(CEM.sphere, [4.0, 4.0], population=40, iterations=30, seed=43)
    check("different seed -> (generally) different path", a != c or CEM.sphere(a) < 1e-6)

    # ---- 7. 1-D optimization ------------------------------------------------------------
    g = lambda p: (p[0] - 7.5) ** 2
    x = CEM.optimize(g, [0.0], std=[10], population=40, iterations=80, seed=6)
    check("1-D minimum found", abs(x[0] - 7.5) < 0.02, f"{x[0]:.3f}")

    # ---- 8. higher dimensions -----------------------------------------------------------
    x = CEM.optimize(CEM.sphere, [2.0] * 6, std=[3] * 6, population=100, iterations=120, seed=7)
    check("6-D sphere minimized", CEM.sphere(x) < 1e-3, f"{CEM.sphere(x):.2e}")

    # ---- 9. benchmark objective values at their known optima ----------------------------
    check("sphere(0) == 0", CEM.sphere([0, 0, 0]) == 0)
    check("rosenbrock(1,1) == 0", abs(CEM.rosenbrock([1, 1])) < 1e-12)
    check("rastrigin(0) == 0", abs(CEM.rastrigin([0, 0])) < 1e-12)

    # ---- 10. elite fraction affects exploration -----------------------------------------
    # a very greedy (tiny elite) run and an exploratory run should both improve on the start
    start_val = CEM.sphere([6.0, 6.0])
    greedy = CEM.optimize(CEM.sphere, [6.0, 6.0], std=[2, 2], population=60, elite_frac=0.05,
                          iterations=50, seed=8)
    explor = CEM.optimize(CEM.sphere, [6.0, 6.0], std=[2, 2], population=60, elite_frac=0.5,
                          iterations=50, seed=8)
    check("both elite fractions improve on the start",
          CEM.sphere(greedy) < start_val and CEM.sphere(explor) < start_val)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

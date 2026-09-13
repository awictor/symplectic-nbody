"""Tests for CMA-ES: reaches known optima on sphere/ellipsoid/Rosenbrock/Rastrigin, monotone best."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cmaes import (  # noqa: E402
    cmaes,
    sphere,
    ellipsoid,
    rosenbrock,
    rastrigin,
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
    # ---- 1. sphere: converges to the origin, value ~ 0 ----------------------------------
    r = cmaes(sphere, [3.0, -2.0, 1.5, 0.7], sigma0=0.5, seed=1)
    check("sphere reaches ~0", r["fx"] < 1e-9, f"fx={r['fx']:.2e}")
    check("sphere optimizer near origin", all(abs(xi) < 1e-4 for xi in r["x"]), f"{r['x']}")

    # ---- 2. ill-conditioned ellipsoid: CMA-ES's specialty -------------------------------
    r = cmaes(ellipsoid, [2.0, 2.0, 2.0, 2.0], sigma0=0.5, seed=2, max_iter=2000)
    check("ellipsoid (cond 1e6) reaches ~0", r["fx"] < 1e-6, f"fx={r['fx']:.2e}")

    # ---- 3. Rosenbrock: the banana valley, min 0 at all-ones ----------------------------
    r = cmaes(rosenbrock, [-1.2, 1.0, 0.5], sigma0=0.4, seed=3, max_iter=3000)
    check("Rosenbrock reaches ~0", r["fx"] < 1e-8, f"fx={r['fx']:.2e}")
    check("Rosenbrock optimizer near (1,1,1)",
          all(abs(xi - 1.0) < 1e-3 for xi in r["x"]), f"{r['x']}")

    # ---- 4. Rastrigin: multimodal (needs a decent start / sigma) ------------------------
    # from a modest start CMA-ES falls into the central basin
    r = cmaes(rastrigin, [0.3, -0.2], sigma0=0.3, seed=5, max_iter=2000)
    check("Rastrigin reaches a low value", r["fx"] < 1.0, f"fx={r['fx']:.2e}")

    # ---- 5. best-so-far history is monotone non-increasing ------------------------------
    r = cmaes(sphere, [5.0, 5.0, 5.0], sigma0=1.0, seed=7)
    hist = r["history"]
    mono = all(hist[i + 1] <= hist[i] + 1e-15 for i in range(len(hist) - 1))
    check("best-so-far is monotone non-increasing", mono)

    # ---- 6. returned fx matches f(x) ----------------------------------------------------
    check("returned fx == f(best_x)", abs(r["fx"] - sphere(r["x"])) < 1e-12)

    # ---- 7. 1-D problem works -----------------------------------------------------------
    r = cmaes(lambda x: (x[0] - 3.0) ** 2, [0.0], sigma0=1.0, seed=9)
    check("1-D minimum at x=3", abs(r["x"][0] - 3.0) < 1e-4, f"{r['x']}")

    # ---- 8. shifted sphere: minimum away from origin ------------------------------------
    target = [1.0, -2.0, 0.5]

    def shifted(x):
        return sum((x[i] - target[i]) ** 2 for i in range(len(x)))

    r = cmaes(shifted, [0.0, 0.0, 0.0], sigma0=0.5, seed=11)
    check("shifted sphere finds the shift",
          all(abs(r["x"][i] - target[i]) < 1e-4 for i in range(3)), f"{r['x']}")

    # ---- 9. bound handling keeps samples in the box -------------------------------------
    lo = [-1.0, -1.0]
    hi = [1.0, 1.0]

    # minimum of this is at (1,1) corner, clamped
    def cornered(x):
        return (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2

    r = cmaes(cornered, [0.0, 0.0], sigma0=0.3, seed=13, bounds=(lo, hi), max_iter=300)
    inbox = all(lo[i] - 1e-9 <= r["x"][i] <= hi[i] + 1e-9 for i in range(2))
    check("bounded run stays in the box", inbox, f"{r['x']}")
    check("bounded optimum at the corner",
          abs(r["x"][0] - 1.0) < 1e-2 and abs(r["x"][1] - 1.0) < 1e-2, f"{r['x']}")

    # ---- 10. multiple seeds all solve the sphere (robustness) ---------------------------
    ok = True
    for s in range(6):
        rr = cmaes(sphere, [4.0, -3.0, 2.0], sigma0=0.5, seed=s + 20)
        if rr["fx"] > 1e-8:
            ok = False
    check("sphere solved across 6 seeds", ok)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

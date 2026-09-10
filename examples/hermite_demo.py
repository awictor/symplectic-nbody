"""Demo: the Hermite scheme vs the others -- 4th order at one force call per step.

RK4 and Forest-Ruth are 4th order but cost 4 and 3 force evaluations per step.
The Hermite predictor-corrector reaches 4th order with a SINGLE force+jerk call.
This table shows the error each method reaches for a fixed force-evaluation
budget on the exact Kepler orbit.

    python examples/hermite_demo.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from systems import two_body_eccentric  # noqa: E402
from kepler import KeplerOrbit  # noqa: E402
from integrators import rk4, forest_ruth  # noqa: E402
from hermite import hermite_step  # noqa: E402

# force evaluations per step for each method
FEVAL = {"rk4": 4, "forest_ruth": 3, "hermite": 1}


def final_error(method, steps):
    k = KeplerOrbit(e=0.5, a=1.0, mu=2.0)
    T = k.period
    s = two_body_eccentric(e=0.5, a=1.0)
    dt = T / steps
    pos, vel = [list(p) for p in s.pos], [list(v) for v in s.vel]
    for _ in range(steps):
        if method == "rk4":
            pos, vel = rk4(pos, vel, s.accel, dt)
        elif method == "forest_ruth":
            pos, vel = forest_ruth(pos, vel, s.accel, dt)
        else:
            pos, vel = hermite_step(pos, vel, s.m, dt, G=s.G)
    tp, _ = k.barycentric(T, 1.0, 1.0)
    return max(abs(pos[i][j] - tp[i][j]) for i in range(2) for j in range(3))


def main():
    print("Accuracy vs force-evaluation budget (exact Kepler orbit, e=0.5)\n")
    budget = 24000  # total force evaluations per method
    print(f"{'method':<14}{'steps':>8}{'f-evals':>10}{'end error':>14}{'order':>8}")
    print("-" * 54)
    for method in ("rk4", "forest_ruth", "hermite"):
        fe = FEVAL[method]
        steps = budget // fe
        err = final_error(method, steps)
        # empirical order from a step-halving at a COARSE count, so the finer
        # run hasn't hit the ~1e-13 round-off floor (which would corrupt the slope)
        e1 = final_error(method, 500)
        e2 = final_error(method, 1000)
        order = math.log(e1 / e2) / math.log(2.0)
        print(f"{method:<14}{steps:>8}{steps*fe:>10}{err:>14.3e}{order:>8.2f}")

    print("\nAll three are 4th order, but for the SAME force-evaluation budget")
    print("Hermite takes 4x as many steps as RK4 (1 call/step vs 4), so it reaches")
    print("a much smaller error. That efficiency is why Hermite runs star clusters.")


if __name__ == "__main__":
    main()

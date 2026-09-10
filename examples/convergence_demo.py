"""Demo: measure each integrator's convergence order against the EXACT Kepler orbit.

Halving the step should cut the error by 2^p, where p is the method's order.
This reads the order straight off the data -- no theory assumed, just measured.

    python examples/convergence_demo.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kepler import KeplerOrbit  # noqa: E402
from systems import two_body_eccentric  # noqa: E402
from integrators import INTEGRATORS  # noqa: E402


def final_error(method, e, steps):
    k = KeplerOrbit(e=e, a=1.0, mu=2.0)
    T = k.period
    s = two_body_eccentric(e=e, a=1.0)
    dt = T / steps
    pos, vel = [list(p) for p in s.pos], [list(v) for v in s.vel]
    integ = INTEGRATORS[method]
    for _ in range(steps):
        pos, vel = integ(pos, vel, s.accel, dt)
    tp, _ = k.barycentric(T, 1.0, 1.0)
    return max(abs(pos[i][j] - tp[i][j]) for i in range(2) for j in range(3))


def main():
    e = 0.5
    step_counts = [500, 1000, 2000, 4000, 8000]
    print(f"Convergence to the exact Kepler orbit (e={e}, one period, G=1)")
    print("error = max position error at t=T vs analytic solution\n")

    for method in ("verlet", "forest_ruth", "rk4"):
        print(f"{method}:")
        print(f"  {'steps':>7}{'error':>14}{'ratio':>9}{'order':>8}")
        prev = None
        for n in step_counts:
            err = final_error(method, e, n)
            if prev is None:
                print(f"  {n:>7}{err:>14.3e}{'--':>9}{'--':>8}")
            else:
                ratio = prev / err
                order = math.log(ratio) / math.log(2.0)
                print(f"  {n:>7}{err:>14.3e}{ratio:>9.2f}{order:>8.2f}")
            prev = err
        print()

    print("verlet doubles-steps -> ~4x smaller error (order 2).")
    print("forest_ruth & rk4 -> ~16x smaller error (order 4).")
    print("Same order, but forest_ruth is symplectic: it ALSO keeps energy")
    print("bounded forever, which rk4 does not (see energy_drift_demo.py).")


if __name__ == "__main__":
    main()

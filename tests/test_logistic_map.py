"""Tests for logistic_map: the period-doubling route to chaos."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logistic_map as lm

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# One step: x' = r x (1-x).
check("step = r x (1-x)", abs(lm.step(0.5, 3.2) - 3.2 * 0.5 * 0.5) < 1e-12)
check("step keeps x in [0,1] for r<=4", 0.0 <= lm.step(0.5, 4.0) <= 1.0)

# Trajectory length and start.
traj = lm.trajectory(0.3, 2.5, 10)
check("trajectory length n+1", len(traj) == 11)
check("trajectory starts at x0", traj[0] == 0.3)

# Fixed point 1 - 1/r; extinction for r<=1.
check("fixed point = 1 - 1/r", abs(lm.fixed_point(2.0) - 0.5) < 1e-12)
check("extinction for r<1", lm.fixed_point(0.8) == 0.0)
# Trajectory converges to the fixed point for 1<r<3.
tail = lm.trajectory(0.2, 2.5, 500)[-1]
check("converges to fixed point (r=2.5)", abs(tail - lm.fixed_point(2.5)) < 1e-4)

# Stability: |2-r|<1 -> 1<r<3.
check("fixed point stable at r=2.5", lm.fixed_point_stable(2.5))
check("unstable at r=3.2", not lm.fixed_point_stable(3.2))
check("unstable at r=0.9", not lm.fixed_point_stable(0.9))

# Attractor / period: 1 in the fixed regime, 2 just past 3, 4 higher, many in chaos.
check("period 1 at r=2.8", lm.period(2.8) == 1)
check("period 2 at r=3.2", lm.period(3.2) == 2)
check("period 4 at r=3.5", lm.period(3.5) == 4)
check("many points in chaos (r=3.9)", lm.period(3.9) > 10)
# Period-3 window near 3.83.
check("period 3 window near 3.83", lm.period(3.83) == 3)

# Lyapunov exponent: negative in periodic regime, positive in chaos.
check("negative Lyapunov in periodic regime (r=3.2)", lm.lyapunov_exponent(3.2) < 0.0)
check("positive Lyapunov in chaos (r=3.9)", lm.lyapunov_exponent(3.9) > 0.0)
check("r=4 fully chaotic, Lyapunov ~ ln 2", abs(lm.lyapunov_exponent(4.0) - math.log(2)) < 0.02)
check("is_chaotic true at r=3.9", lm.is_chaotic(3.9))
check("is_chaotic false at r=3.2", not lm.is_chaotic(3.2))

# Feigenbaum constant.
check("Feigenbaum delta ~4.669", abs(lm.FEIGENBAUM_DELTA - 4.6692) < 0.001)
check("chaos onset ~3.5699", abs(lm.CHAOS_ONSET - 3.5699) < 0.001)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all logistic_map tests passed")

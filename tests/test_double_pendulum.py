"""Tests for double_pendulum: chaotic mechanics and energy conservation."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import double_pendulum as dp

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# At rest hanging straight down (theta=0, omega=0), nothing moves.
rest = (0.0, 0.0, 0.0, 0.0)
d = dp.derivatives(rest)
check("straight-down rest is an equilibrium", all(abs(v) < 1e-12 for v in d))

# A displaced pendulum has nonzero angular acceleration.
d2 = dp.derivatives((0.5, 0.0, 0.5, 0.0))
check("displaced pendulum accelerates", abs(d2[1]) > 0.0 or abs(d2[3]) > 0.0)

# Trajectory length/start.
traj = dp.trajectory((1.0, 0.0, 1.0, 0.0), 0.01, 100)
check("trajectory length n+1", len(traj) == 101)
check("trajectory starts at state0", traj[0] == (1.0, 0.0, 1.0, 0.0))

# Energy conservation: with a small enough step over a short time, RK4 keeps the total energy
# nearly constant (a stringent physics+solver check; the chaotic long-time drift of a
# non-symplectic scheme is a separate, expected numerical effect).
state0 = (1.0, 0.0, 0.5, 0.0)
E0 = dp.total_energy(state0)
tr = dp.trajectory(state0, 1e-4, 2000)          # 0.2 s at a fine step
E_end = dp.total_energy(tr[-1])
check("energy conserved to <1% over a short run", abs(E_end - E0) < 1e-2 * abs(E0))
# Over a very fine, brief run it is conserved to high precision.
tr2 = dp.trajectory(state0, 1e-5, 1000)
check("energy conserved to <1e-5 at fine step",
      abs(dp.total_energy(tr2[-1]) - E0) < 1e-5 * abs(E0))

# Positions: at rest both bobs hang straight down.
x1, y1, x2, y2 = dp.positions(rest)
check("rest bob1 at (0,-l1)", abs(x1) < 1e-12 and abs(y1 + 1.0) < 1e-12)
check("rest bob2 at (0,-l1-l2)", abs(x2) < 1e-12 and abs(y2 + 2.0) < 1e-12)
# Horizontal release: first bob out to the side.
xh = dp.positions((math.pi / 2, 0, math.pi / 2, 0))
check("horizontal release bob1 at (l1,0)", abs(xh[0] - 1.0) < 1e-9 and abs(xh[1]) < 1e-9)

# Total energy rises with amplitude (more raised = more PE).
check("higher release, more energy",
      dp.total_energy((2.0, 0, 2.0, 0)) > dp.total_energy((0.3, 0, 0.3, 0)))

# Chaos: a high-energy start diverges much faster than a tiny-amplitude (near-linear) one.
lam_chaos = dp.divergence_rate((math.pi / 2, 0, math.pi / 2, 0), dt=0.005, n=4000)
lam_small = dp.divergence_rate((0.05, 0, 0.05, 0), dt=0.005, n=4000)
check("high-energy start diverges (lambda>0)", lam_chaos > 0.1)
check("chaotic start diverges faster than near-linear one", lam_chaos > 1.5 * lam_small)

# Two nearby high-energy starts separate over time: the gap at the end is far larger than the
# 1e-4 it started at (exponential growth, the butterfly effect).
a = dp.trajectory((math.pi / 2, 0, math.pi / 2, 0), 0.005, 1200)
b = dp.trajectory((math.pi / 2 + 1e-4, 0, math.pi / 2, 0), 0.005, 1200)
def gap(i):
    return math.sqrt(sum((a[i][k] - b[i][k]) ** 2 for k in range(4)))
check("separation grows well beyond its initial 1e-4", gap(len(a) - 1) > 10 * 1e-4)
check("separation larger at end than start", gap(len(a) - 1) > gap(1))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all double_pendulum tests passed")

"""Tests for lorenz: the butterfly strange attractor."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lorenz

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Derivatives vanish at a fixed point (the origin).
dx, dy, dz = lorenz.derivatives((0.0, 0.0, 0.0))
check("origin is a fixed point", abs(dx) + abs(dy) + abs(dz) < 1e-12)
# Derivatives at C+ also vanish.
c = math.sqrt(lorenz.BETA * (lorenz.RHO - 1.0))
d = lorenz.derivatives((c, c, lorenz.RHO - 1.0))
check("convection point C+ is fixed", sum(abs(v) for v in d) < 1e-9)

# Trajectory length and start.
traj = lorenz.trajectory((1.0, 1.0, 1.0), 0.01, 100)
check("trajectory length n+1", len(traj) == 101)
check("trajectory starts at state0", traj[0] == (1.0, 1.0, 1.0))

# The attractor is bounded (does not blow up); z stays positive-ish, orbits ~|x|<25.
long_traj = lorenz.trajectory((1.0, 1.0, 1.0), 0.01, 5000)[2000:]
xs = [p[0] for p in long_traj]
zs = [p[2] for p in long_traj]
check("x bounded on attractor", all(abs(xv) < 30 for xv in xs))
check("z bounded and positive-ish", all(-5 < zv < 60 for zv in zs))
check("orbit visits both lobes (x changes sign)", min(xs) < -1 and max(xs) > 1)

# Volume contraction div F = -(sigma+1+beta).
check("volume contraction = -(sigma+1+beta)",
      abs(lorenz.volume_contraction_rate() - (-(10.0 + 1.0 + 8.0 / 3.0))) < 1e-9)
check("dissipative (negative divergence)", lorenz.volume_contraction_rate() < 0.0)

# Fixed points: origin always; two more for rho>1.
check("origin always a fixed point", (0.0, 0.0, 0.0) in lorenz.fixed_points())
check("three fixed points for rho>1", len(lorenz.fixed_points()) == 3)
check("only origin for rho<1", len(lorenz.fixed_points(rho=0.5)) == 1)
# C+/- coordinates.
fps = lorenz.fixed_points()
check("convection points at z = rho-1", any(abs(p[2] - (lorenz.RHO - 1.0)) < 1e-9 for p in fps[1:]))

# Largest Lyapunov exponent: positive (~0.9) for classic parameters.
lam = lorenz.largest_lyapunov(n=8000)
check("classic Lyapunov positive (chaos)", lam > 0.0)
check("classic Lyapunov ~0.9", 0.7 < lam < 1.1)

# Below rho=1 the origin is globally stable -> non-positive Lyapunov, trajectory decays to 0.
tail = lorenz.trajectory((1.0, 1.0, 1.0), 0.01, 3000, rho=0.5)[-1]
check("decays to origin for rho<1", sum(abs(v) for v in tail) < 0.1)

# RK4 conserves nothing but should be reversible-ish over one small step (sanity: finite).
s1 = lorenz.rk4_step((1.0, 1.0, 1.0), 0.01)
check("one step gives finite state", all(math.isfinite(v) for v in s1))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all lorenz tests passed")

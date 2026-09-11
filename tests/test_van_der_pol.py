"""Tests for van_der_pol: the self-sustaining limit-cycle oscillator."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import van_der_pol as vp

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Vector field: dx = v; dv = mu(1-x^2)v - x.
dx, dv = vp.derivatives((0.5, 1.0), 2.0)
check("dx = v", abs(dx - 1.0) < 1e-12)
check("dv = mu(1-x^2)v - x", abs(dv - (2.0 * (1 - 0.25) * 1.0 - 0.5)) < 1e-12)
# Origin is a fixed point (unstable, but a fixed point).
check("origin is a fixed point", vp.derivatives((0.0, 0.0), 1.0) == (0.0, 0.0))

# Trajectory length/start.
tr = vp.trajectory((0.1, 0.0), 0.01, 100, mu=1.0)
check("trajectory length n+1", len(tr) == 101)
check("trajectory starts at state0", tr[0] == (0.1, 0.0))

# From a tiny start the amplitude GROWS to the limit cycle (~2), not decays.
tr2 = vp.trajectory((0.01, 0.0), 0.01, 3000, mu=1.0)
early = max(abs(p[0]) for p in tr2[:200])
late = max(abs(p[0]) for p in tr2[-500:])
check("amplitude grows from tiny start", late > early)

# Limit-cycle amplitude ~2 for a range of mu.
for mu in (0.5, 1.0, 3.0):
    amp = vp.limit_cycle_amplitude(mu)
    check(f"limit-cycle amplitude ~2 (mu={mu})", 1.8 < amp < 2.3)

# Different starts converge to the SAME limit-cycle amplitude (forgets initial conditions).
a1 = vp.limit_cycle_amplitude(1.0)
# start big instead of small -> same cycle
s = (4.0, 0.0)
for _ in range(4000):
    s = vp.rk4_step(s, 0.01, 1.0)
peak_big = 0.0
for _ in range(2000):
    s = vp.rk4_step(s, 0.01, 1.0)
    peak_big = max(peak_big, abs(s[0]))
check("large start converges to same cycle", abs(peak_big - a1) < 0.1)

# Self-sustaining only for mu>0.
check("mu>0 self-sustaining", vp.is_self_sustaining(1.0))
check("mu=0 not self-sustaining", not vp.is_self_sustaining(0.0))
check("mu<0 not self-sustaining", not vp.is_self_sustaining(-1.0))
# mu<0 damps to rest.
sd = (1.0, 0.0)
for _ in range(4000):
    sd = vp.rk4_step(sd, 0.01, -0.5)
check("mu<0 decays to rest", abs(sd[0]) < 0.05 and abs(sd[1]) < 0.05)

# Relaxation period ~1.614 mu for large mu.
check("relaxation period ~1.614 mu", abs(vp.relaxation_period(10.0) - 16.14) < 0.1)
check("period grows with mu", vp.relaxation_period(20.0) > vp.relaxation_period(10.0))

# Small-mu period is near 2 pi (nearly harmonic).
T = vp.measured_period(0.1)
check("small-mu period ~2 pi", abs(T - 2 * math.pi) < 0.3)
# Large-mu measured period grows well past the harmonic 2 pi (relaxation regime); the
# 1.614*mu formula is the mu->inf asymptote, so at mu=5 the true period is still larger.
T_big = vp.measured_period(5.0)
check("large-mu period much longer than 2 pi", T_big > 6.0)
check("large-mu period scales up (>= relaxation estimate)", T_big > vp.relaxation_period(5.0))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all van_der_pol tests passed")

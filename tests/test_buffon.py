"""Tests for buffon: needle-drop geometric probability and Monte Carlo pi."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import buffon

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Crossing probability P = 2L/(pi d). Equal L and d -> 2/pi ~ 0.6366.
check("P = 2/pi for L = d", abs(buffon.crossing_probability(1.0, 1.0) - 2.0 / math.pi) < 1e-9)
check("P = 2L/(pi d)", abs(buffon.crossing_probability(0.5, 1.0) - 1.0 / math.pi) < 1e-9)
# Longer needle (still <= d) crosses more often.
check("longer needle crosses more", buffon.crossing_probability(0.8, 1.0) > buffon.crossing_probability(0.4, 1.0))
# Wider spacing crosses less.
check("wider spacing crosses less", buffon.crossing_probability(0.5, 2.0) < buffon.crossing_probability(0.5, 1.0))
# Long-needle formula rejected.
try:
    buffon.crossing_probability(2.0, 1.0)
    check("long needle raises", False)
except ValueError:
    check("long needle raises", True)

# estimate_pi inverts the probability: if C = P*N exactly, estimate returns pi.
L, d, N = 1.0, 1.0, 100000
C_exact = buffon.crossing_probability(L, d) * N       # expected crossings
check("estimate_pi recovers pi from exact crossings",
      abs(buffon.estimate_pi(L, d, N, C_exact) - math.pi) < 1e-6)
check("no crossings gives inf", math.isinf(buffon.estimate_pi(L, d, N, 0)))
# More crossings -> smaller pi estimate (inverse relation).
check("more crossings, smaller pi estimate",
      buffon.estimate_pi(L, d, N, 70000) < buffon.estimate_pi(L, d, N, 60000))

# needles_for_accuracy: 1/error^2 scaling.
check("1% error needs ~1e4 drops", 8000 < buffon.needles_for_accuracy(0.01) < 12000)
check("0.1% needs ~1e6", 8e5 < buffon.needles_for_accuracy(0.001) < 1.2e6)
check("tighter accuracy needs more drops",
      buffon.needles_for_accuracy(0.005) > buffon.needles_for_accuracy(0.01))

# Monte Carlo error ~ 1/sqrt(N): quadruple N halves the error.
check("error ~ 1/sqrt(N)",
      abs(buffon.monte_carlo_error(400) - buffon.monte_carlo_error(100) / 2.0) < 1e-9)

# Simulation: crossing fraction near 2/pi, pi estimate near 3.14 for many drops.
crosses, pi_est = buffon.simulate(1.0, 1.0, 100000, seed=1)
frac = crosses / 100000
check("simulated crossing fraction ~2/pi", abs(frac - 2.0 / math.pi) < 0.02)
check("simulated pi within 3% of pi", abs(pi_est - math.pi) < 0.1)
# Reproducible with the same seed.
c1, _ = buffon.simulate(1.0, 1.0, 5000, seed=7)
c2, _ = buffon.simulate(1.0, 1.0, 5000, seed=7)
check("simulation reproducible", c1 == c2)
# Different needle length changes the crossing fraction as ~L.
c_short, _ = buffon.simulate(0.5, 1.0, 100000, seed=1)
check("half-length needle roughly halves crossings",
      abs(c_short / crosses - 0.5) < 0.05)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all buffon tests passed")

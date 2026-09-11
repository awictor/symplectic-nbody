"""Tests for kuramoto: coupled-oscillator synchronization."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import kuramoto as ku

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Order parameter: 1 for identical phases, ~0 for evenly spread.
check("r=1 for aligned phases", abs(ku.order_parameter([0.5, 0.5, 0.5, 0.5]) - 1.0) < 1e-12)
even = [2 * math.pi * i / 8 for i in range(8)]
check("r~0 for evenly spread phases", ku.order_parameter(even) < 1e-9)
# Two antiphase oscillators cancel.
check("r=0 for antiphase pair", abs(ku.order_parameter([0.0, math.pi])) < 1e-12)
# Partial clustering gives intermediate r.
check("r intermediate for partial cluster", 0.0 < ku.order_parameter([0.0, 0.3, 0.6]) < 1.0)
# r in [0,1].
check("r in [0,1]", 0.0 <= ku.order_parameter([0.1, 1.0, 2.0, 3.0]) <= 1.0)

# Mean phase of aligned phases is that phase.
check("mean phase of aligned = phase", abs(ku.mean_phase([0.7, 0.7, 0.7]) - 0.7) < 1e-9)

# Derivatives: with zero coupling, each just runs at its natural frequency.
d = ku.derivatives([0.0, 1.0], [2.0, -1.0], 0.0)
check("zero coupling -> natural frequencies", abs(d[0] - 2.0) < 1e-12 and abs(d[1] + 1.0) < 1e-12)
# Coupling pulls a lagging oscillator forward.
d2 = ku.derivatives([0.0, 1.0], [0.0, 0.0], 2.0)
check("coupling pulls phases together", d2[0] > 0.0 and d2[1] < 0.0)

# Critical coupling: Lorentzian K_c = 2 gamma.
check("Lorentzian K_c = 2 gamma", abs(ku.critical_coupling_lorentzian(0.5) - 1.0) < 1e-12)
check("Gaussian K_c = sigma sqrt(8/pi)",
      abs(ku.critical_coupling_gaussian(1.0) - math.sqrt(8.0 / math.pi)) < 1e-9)
check("wider spread needs stronger coupling",
      ku.critical_coupling_lorentzian(1.0) > ku.critical_coupling_lorentzian(0.5))

# Simulation: weak coupling stays incoherent, strong coupling synchronizes.
r_weak = ku.simulate(40, k=0.2, omega_spread=1.0, seed=1)
r_strong = ku.simulate(40, k=6.0, omega_spread=1.0, seed=1)
check("weak coupling incoherent (r small)", r_weak < 0.4)
check("strong coupling synchronizes (r high)", r_strong > 0.8)
check("synchrony rises with coupling", r_strong > r_weak)

# Monotone-ish: intermediate coupling gives intermediate synchrony.
r_mid = ku.simulate(40, k=2.0, omega_spread=1.0, seed=1)
check("intermediate coupling between weak and strong", r_weak <= r_mid <= r_strong + 1e-9)

# Zero coupling with a frequency spread stays incoherent.
check("zero coupling incoherent", ku.simulate(40, k=0.0, omega_spread=1.0, seed=1) < 0.5)
# Identical oscillators (no spread) synchronize even at modest coupling.
check("no-spread population synchronizes", ku.simulate(30, k=1.0, omega_spread=0.0, seed=1) > 0.9)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all kuramoto tests passed")

"""Tests for kutta_joukowski: lift from circulation and the Magnus effect."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import kutta_joukowski as kj

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Kutta-Joukowski lift L' = rho U Gamma.
check("L' = rho U Gamma", abs(kj.kutta_joukowski_lift(10.0, 50.0, 1.225) - 1.225 * 50.0 * 10.0) < 1e-6)
check("no lift without circulation", kj.kutta_joukowski_lift(0.0, 50.0) == 0.0)

# Thin-airfoil lift-slope: c_l = 2 pi alpha.
check("c_l = 2 pi alpha", abs(kj.lift_coefficient(0.1) - 2.0 * math.pi * 0.1) < 1e-12)
# ~0.11 per degree.
check("lift-slope ~0.11 per degree",
      abs(kj.lift_coefficient(math.radians(1.0)) - 0.1097) < 1e-3)
# 5 degrees -> c_l ~ 0.548.
check("c_l ~0.55 at 5 deg", abs(kj.lift_coefficient(math.radians(5.0)) - 0.548) < 0.01)

# Circulation and lift are consistent: L' from Gamma equals c_l-based lift per unit span.
U, c, alpha = 50.0, 1.5, math.radians(4.0)
Gamma = kj.thin_airfoil_circulation(U, c, alpha)
L_circ = kj.kutta_joukowski_lift(Gamma, U)
L_cl = kj.lift_force(kj.lift_coefficient(alpha), U, c)   # area = c * 1 m span
check("circulation lift = coefficient lift", abs(L_circ - L_cl) < 1e-6)

# Lift force scales as U^2 and area.
L1 = kj.lift_force(0.5, 50.0, 20.0)
check("lift scales as U^2", abs(kj.lift_force(0.5, 100.0, 20.0) - 4.0 * L1) < 1e-6)
check("lift scales with area", abs(kj.lift_force(0.5, 50.0, 40.0) - 2.0 * L1) < 1e-6)

# A light aircraft: 20 m^2 wing, c_l=0.5, 50 m/s -> lift ~ 15 kN (holds ~1.5 tonnes).
L = kj.lift_force(0.5, 50.0, 20.0)
check("light-aircraft lift ~15 kN", 12000.0 < L < 18000.0)

# Magnus circulation Gamma = 2 pi r^2 omega.
check("Magnus circulation 2 pi r^2 omega",
      abs(kj.magnus_circulation(0.1, 100.0) - 2.0 * math.pi * 0.01 * 100.0) < 1e-9)
# Faster spin -> more circulation -> more side force.
check("faster spin, larger Magnus force",
      kj.magnus_force(0.033, 200.0, 30.0, 1.0) > kj.magnus_force(0.033, 100.0, 30.0, 1.0))

# A spinning ball feels a real sideways force (tennis ball scale).
F = kj.magnus_force(0.033, 150.0, 25.0, 0.066)   # r=3.3cm, 150 rad/s, 25 m/s, length~diameter
check("spinning ball Magnus force positive", F > 0.0)

# Induced drag falls with aspect ratio.
check("induced drag falls with AR",
      kj.induced_drag_coefficient(0.5, 20.0) < kj.induced_drag_coefficient(0.5, 5.0))
# c_di = c_l^2 / (pi AR e).
check("c_di = c_l^2/(pi AR e)",
      abs(kj.induced_drag_coefficient(0.6, 8.0) - 0.36 / (math.pi * 8.0)) < 1e-9)
# Induced drag scales as c_l^2.
check("induced drag scales as c_l^2",
      abs(kj.induced_drag_coefficient(1.0, 8.0) - 4.0 * kj.induced_drag_coefficient(0.5, 8.0)) < 1e-9)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all kutta_joukowski tests passed")

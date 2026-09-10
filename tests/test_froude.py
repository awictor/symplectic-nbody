"""Tests for froude: free-surface flow regimes, hull speed, hydraulic jump."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import froude as fr

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Shallow-water wave speed: 1 m deep -> ~3.13 m/s.
check("wave speed sqrt(g h): 1 m -> ~3.13 m/s", abs(fr.wave_speed(1.0) - 3.132) < 0.01)
check("wave speed quadruples depth -> doubles", abs(fr.wave_speed(4.0) - 2 * fr.wave_speed(1.0)) < 1e-9)

# Froude number = U / sqrt(g h).
check("Fr = U / sqrt(g h)", abs(fr.froude_number(3.132, 1.0) - 1.0) < 1e-3)

# Flow regimes.
check("slow deep flow is subcritical", fr.flow_regime(fr.froude_number(0.5, 1.0)) == "subcritical")
check("fast shallow flow is supercritical", fr.flow_regime(fr.froude_number(5.0, 0.2)) == "supercritical")
check("Fr = 1 is critical", fr.flow_regime(1.0) == "critical")

# Critical depth: q^2/g ^ 1/3, and Fr = 1 there. For q = U*h, check self-consistency.
q = 2.0                    # m^2/s
hc = fr.critical_depth(q)
U_c = q / hc
check("Fr = 1 at critical depth", abs(fr.froude_number(U_c, hc) - 1.0) < 1e-6)

# Hull Froude and hull speed: the wall sits at Fr ~ 0.4028.
L = 10.0                   # 10 m waterline
Vh = fr.hull_speed(L)
check("hull speed at Fr ~ 0.4028", abs(fr.hull_froude(Vh, L) - 0.4028) < 1e-6)
# 1.34 sqrt(L_ft) knots rule: 10 m = 32.8 ft -> ~7.7 kn -> ~3.95 m/s.
check("hull speed ~ 1.34 sqrt(Lft) kn rule", abs(Vh - 3.99) < 0.15)

# Hydraulic jump: supercritical upstream -> deeper downstream.
h2 = fr.conjugate_depth(0.1, 3.0)      # Fr1 = 3
check("jump deepens the stream", h2 > 0.1)
# Belanger: h2/h1 = 0.5(sqrt(1+8*9)-1) = 0.5(sqrt(73)-1) ~ 3.772
check("Belanger conjugate depth ratio", abs(h2 / 0.1 - 0.5 * (math.sqrt(73.0) - 1.0)) < 1e-9)
# A critical upstream flow (Fr1 = 1) makes no jump (h2 = h1).
check("no jump at Fr1 = 1", abs(fr.conjugate_depth(0.1, 1.0) - 0.1) < 1e-9)

# Momentum check: the jump conserves momentum flux M = q^2/(g h) + h^2/2 across it.
Fr1, h1 = 3.0, 0.1
c1 = fr.wave_speed(h1)
U1 = Fr1 * c1
q_jump = U1 * h1
h2j = fr.conjugate_depth(h1, Fr1)
def momentum(h):
    return q_jump ** 2 / (fr.G_EARTH * h) + h * h / 2.0
check("hydraulic jump conserves momentum flux",
      abs(momentum(h1) - momentum(h2j)) < 1e-4 * momentum(h1))

# Kelvin wake half-angle is arcsin(1/3) ~ 19.47 deg, speed-independent.
check("Kelvin wake half-angle ~19.47 deg", abs(fr.kelvin_wake_half_angle() - 19.4712) < 1e-3)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all froude tests passed")

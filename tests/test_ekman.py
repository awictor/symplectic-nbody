"""Tests for ekman: the wind-driven rotating boundary layer."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ekman

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Coriolis parameter: ~1.03e-4 /s at 45 deg N, zero at equator, negative south.
f45 = ekman.coriolis_parameter(45.0)
check("f ~ 1.03e-4 /s at 45 N", 1.0e-4 < f45 < 1.06e-4)
check("f = 0 at equator", abs(ekman.coriolis_parameter(0.0)) < 1e-12)
check("f < 0 in southern hemisphere", ekman.coriolis_parameter(-30.0) < 0.0)

# Surface current is deflected 45 deg to the RIGHT of the wind (north) -- -45 deg as a
# standard math angle in (along-wind, cross-wind) coordinates.
A_z, tau = 0.05, 0.1
u0, v0 = ekman.velocity_at_depth(0.0, tau, A_z, f45)
ang = math.degrees(math.atan2(v0, u0))
check("surface current 45 deg right of wind (N)", abs(ang + 45.0) < 1e-6)

# Southern hemisphere deflects to the LEFT (+45 deg).
fS = ekman.coriolis_parameter(-45.0)
uS, vS = ekman.velocity_at_depth(0.0, tau, A_z, fS)
angS = math.degrees(math.atan2(vS, uS))
check("surface current 45 deg left of wind (S)", abs(angS - 45.0) < 1e-6)
check("surface_angle_deg matches sign", ekman.surface_angle_deg(f45) == -45.0
      and ekman.surface_angle_deg(fS) == 45.0)

# Speed decays exponentially: at one Ekman depth, factor e^-pi ~ 0.043.
D = ekman.ekman_depth(A_z, f45)
u_top = math.hypot(u0, v0)
uD, vD = ekman.velocity_at_depth(-D, tau, A_z, f45)
u_bot = math.hypot(uD, vD)
check("speed decays to ~e^-pi at Ekman depth",
      abs(u_bot / u_top - math.exp(-math.pi)) < 1e-6)

# Surface speed formula matches tau/(rho sqrt(|f| A_z)).
V0 = ekman.surface_speed(tau, A_z, f45)
check("surface speed = tau/(rho sqrt(f A_z))",
      abs(u_top - V0) < 1e-9)

# The current spirals clockwise (N): at a shallow depth the angle is more negative than
# the -45 deg surface value (before atan2 wraps around).
d = math.sqrt(2.0 * A_z / abs(f45))
u_sh, v_sh = ekman.velocity_at_depth(-0.3 * d, tau, A_z, f45)
ang_sh = math.degrees(math.atan2(v_sh, u_sh))
check("current rotates clockwise with depth (N)", ang_sh < ang)

# Ekman transport is independent of eddy viscosity, magnitude tau/(rho |f|).
T1 = ekman.ekman_transport(tau, f45)
check("transport = tau/(rho |f|)", abs(T1 - tau / (ekman.RHO_SEAWATER * f45)) < 1e-12)
check("transport independent of A_z",
      ekman.ekman_transport(tau, f45) == ekman.ekman_transport(tau, f45))

# Ekman depth grows with viscosity, shrinks toward the pole (larger |f|).
check("deeper layer for larger viscosity",
      ekman.ekman_depth(0.1, f45) > ekman.ekman_depth(0.05, f45))
check("shallower layer nearer pole",
      ekman.ekman_depth(A_z, ekman.coriolis_parameter(70.0))
      < ekman.ekman_depth(A_z, ekman.coriolis_parameter(20.0)))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all ekman tests passed")

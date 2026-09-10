"""Tests for womersley: pulsatile flow and the heartbeat lag."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import womersley as wo

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Blood: nu ~ 3.5e-6 m^2/s. Aorta R ~ 0.011 m at ~60 bpm -> alpha ~ 12-16.
NU_BLOOD = 3.5e-6
alpha_aorta = wo.womersley_from_heart_rate(0.011, 60.0, NU_BLOOD)
check("aorta alpha ~15", 10.0 < alpha_aorta < 20.0)

# A capillary (R ~ 4 microns) has alpha << 1 -- quasi-steady.
alpha_cap = wo.womersley_from_heart_rate(4e-6, 60.0, NU_BLOOD)
check("capillary alpha << 1", alpha_cap < 0.01)
check("capillary is quasi-steady", wo.is_quasi_steady(alpha_cap))
check("aorta is not quasi-steady", not wo.is_quasi_steady(alpha_aorta))

# alpha = R sqrt(omega/nu): scales with R and sqrt(omega).
check("alpha linear in radius",
      abs(wo.womersley_number(0.02, 6.0, NU_BLOOD) - 2.0 * wo.womersley_number(0.01, 6.0, NU_BLOOD)) < 1e-9)
check("alpha scales as sqrt(omega)",
      abs(wo.womersley_number(0.01, 24.0, NU_BLOOD) - 2.0 * wo.womersley_number(0.01, 6.0, NU_BLOOD)) < 1e-9)

# alpha = R / penetration_depth.
omega = 2.0 * math.pi
delta = wo.penetration_depth(omega, NU_BLOOD)
check("alpha = R / penetration depth",
      abs(wo.womersley_number(0.01, omega, NU_BLOOD) - 0.01 / delta) < 1e-9)
# Higher frequency -> thinner penetration depth.
check("faster oscillation, thinner Stokes layer",
      wo.penetration_depth(4 * omega, NU_BLOOD) < wo.penetration_depth(omega, NU_BLOOD))

# Phase lag: ~0 for small alpha, approaching pi/2 for large alpha.
check("phase lag ~0 for small alpha", wo.phase_lag(0.1) < 0.05)
check("phase lag approaches pi/2 for large alpha", wo.phase_lag(50.0) > 1.5)
check("phase lag increases with alpha", wo.phase_lag(10.0) > wo.phase_lag(1.0))
check("phase lag bounded by pi/2", wo.phase_lag(1e6) < math.pi / 2.0)

# Poiseuille flow: Q = pi R^4 dP/dx / (8 mu), scales as R^4.
q1 = wo.poiseuille_flow(0.01, 100.0, 3.5e-3)
check("Poiseuille Q scales as R^4",
      abs(wo.poiseuille_flow(0.02, 100.0, 3.5e-3) - 16.0 * q1) < 1e-6 * 16.0 * q1)
check("Poiseuille Q positive", q1 > 0.0)

# Moens-Korteweg pulse-wave speed: aorta E~5e5 Pa, h~2mm, R~11mm -> ~6-10 m/s.
c = wo.pulse_wave_speed(5e5, 0.002, 0.011)
check("aortic pulse-wave speed 4-12 m/s", 4.0 < c < 12.0)
# Stiffer artery -> faster pulse wave (arterial aging).
check("stiffer artery faster pulse wave",
      wo.pulse_wave_speed(1e6, 0.002, 0.011) > c)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all womersley tests passed")

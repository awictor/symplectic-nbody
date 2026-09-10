"""Tests for surface_tension: Jurin's law and Young-Laplace, checked against water."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import surface_tension as st

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Water in a 1 mm-diameter tube (r = 0.5 mm) rises ~1.5 cm -- the textbook value.
h = st.capillary_rise(st.GAMMA_WATER, 0.5e-3)
check("water rises ~1.5 cm in 1 mm tube", 0.025 < h < 0.035)

# Rise scales as 1/r: halve the radius, double the rise.
h_half = st.capillary_rise(st.GAMMA_WATER, 0.25e-3)
check("rise doubles when radius halves", abs(h_half - 2.0 * h) < 1e-9)

# A 1 micron pore lifts water many metres (root-uptake regime).
h_pore = st.capillary_rise(st.GAMMA_WATER, 1e-6)
check("1 micron pore lifts water > 10 m", h_pore > 10.0)

# Mercury (contact angle ~140 deg) is depressed, not raised.
h_hg = st.capillary_rise(st.GAMMA_MERCURY, 0.5e-3, rho=13534.0,
                         contact_angle_rad=math.radians(140.0))
check("mercury is depressed (negative rise)", h_hg < 0.0)

# radius_for_rise inverts capillary_rise.
r_back = st.radius_for_rise(st.GAMMA_WATER, h)
check("radius_for_rise inverts capillary_rise", abs(r_back - 0.5e-3) < 1e-9)

# Bubble overpressure is twice the droplet's at the same radius (two films).
r = 1e-3
check("bubble pressure = 2x droplet pressure",
      abs(st.bubble_pressure(st.GAMMA_WATER, r)
          - 2.0 * st.droplet_pressure(st.GAMMA_WATER, r)) < 1e-12)

# Smaller droplet has higher internal pressure (Laplace 1/r).
check("smaller droplet higher pressure",
      st.droplet_pressure(st.GAMMA_WATER, 1e-4)
      > st.droplet_pressure(st.GAMMA_WATER, 1e-3))

# A 1 mm droplet's overpressure is ~146 Pa (2*0.0728/1e-3).
dp = st.droplet_pressure(st.GAMMA_WATER, 1e-3)
check("1 mm droplet overpressure ~146 Pa", 140.0 < dp < 152.0)

# Surface energy is gamma * area.
check("surface energy = gamma*area",
      abs(st.surface_energy(st.GAMMA_WATER, 2.0) - 2.0 * st.GAMMA_WATER) < 1e-12)

# Water strider: 4 legs, 5 mm contact each -> supports well over a 10 mg insect.
F = st.max_supported_weight(st.GAMMA_WATER, 4 * 5e-3)
check("surface tension holds a 10 mg strider", F > 10e-6 * st.G_EARTH)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all surface_tension tests passed")

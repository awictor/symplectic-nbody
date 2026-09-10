"""Tests for capillary: the gravity-vs-surface-tension length and Bond/Weber numbers."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import capillary as cap

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Water's capillary length is ~2.7 mm.
lc = cap.capillary_length(cap.GAMMA_WATER)
check("water capillary length ~2.7 mm", 0.0025 < lc < 0.0029)

# Bond number = (L / l_c)^2, so Bo = 1 exactly at L = l_c.
check("Bond = 1 at L = capillary length",
      abs(cap.bond_number(lc, cap.GAMMA_WATER) - 1.0) < 1e-9)
check("Bond = (L/l_c)^2",
      abs(cap.bond_number(2 * lc, cap.GAMMA_WATER) - 4.0) < 1e-9)

# Small drop (1 mm < l_c): surface tension dominates, Bo < 1.
check("1 mm drop is surface-tension-dominated (Bo<1)",
      cap.bond_number(1e-3, cap.GAMMA_WATER) < 1.0)
# Big puddle (1 cm > l_c): gravity dominates, Bo > 1.
check("1 cm puddle is gravity-dominated (Bo>1)",
      cap.bond_number(1e-2, cap.GAMMA_WATER) > 1.0)

# Weber number = rho v^2 L / gamma.
check("Weber = rho v^2 L / gamma",
      abs(cap.weber_number(2.0, 1e-3, cap.GAMMA_WATER)
          - cap.RHO_WATER * 4.0 * 1e-3 / cap.GAMMA_WATER) < 1e-6)

# Breakup: a slow 2 mm drop stays intact; a fast one fragments.
check("slow drop stays intact", not cap.droplet_breaks_up(0.1, 2e-3, cap.GAMMA_WATER))
check("fast drop breaks up", cap.droplet_breaks_up(5.0, 2e-3, cap.GAMMA_WATER))

# breakup_velocity gives exactly We = We_crit.
vb = cap.breakup_velocity(2e-3, cap.GAMMA_WATER)
check("breakup velocity hits critical Weber",
      abs(cap.weber_number(vb, 2e-3, cap.GAMMA_WATER) - cap.WE_CRIT) < 1e-6)

# Max puddle depth for a fully non-wetting liquid is 2 l_c (~5.4 mm for water).
h = cap.max_puddle_depth(cap.GAMMA_WATER, contact_angle_rad=math.pi)
check("max puddle depth = 2 l_c", abs(h - 2.0 * lc) < 1e-9)
check("water puddle max depth ~5.4 mm", 0.005 < h < 0.0058)
# Wetting liquid (small contact angle) puddles thinner.
check("wetting liquid puddles thinner",
      cap.max_puddle_depth(cap.GAMMA_WATER, contact_angle_rad=math.radians(20.0)) < h)

# Rayleigh-Plateau: drops spaced ~9 radii apart.
check("Rayleigh-Plateau spacing ~9 radii",
      abs(cap.rayleigh_plateau_spacing(1.0) - 9.01) < 0.1)

# Mercury has a larger capillary length than water (higher gamma, but much denser).
# gamma_Hg ~ 0.487, rho ~ 13534 -> l_c ~ 1.9 mm, smaller than water's.
lc_hg = cap.capillary_length(0.487, rho=13534.0)
check("mercury capillary length ~1.9 mm", 0.0017 < lc_hg < 0.0021)

# Lower gravity (Moon) raises the capillary length.
check("weaker gravity raises capillary length",
      cap.capillary_length(cap.GAMMA_WATER, g=1.62) > lc)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all capillary tests passed")

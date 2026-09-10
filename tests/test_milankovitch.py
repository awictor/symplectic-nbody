"""Tests for milankovitch: orbital insolation forcing of the ice ages."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import milankovitch as mk

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


PHI65 = math.radians(65.0)

# June-solstice 65N insolation is the textbook ~480-490 W/m^2 (present orbit).
q65 = mk.summer_solstice_insolation(PHI65, e=0.0167,
                                    longitude_perihelion=math.radians(283.0))
check("65N June solstice ~480-490 W/m^2", 470.0 < q65 < 500.0)

# Declination: +eps at June solstice, -eps at December, 0 at equinoxes.
check("declination = +obliquity at June solstice",
      abs(mk.declination(math.pi / 2.0) - mk.OBLIQUITY_NOW) < 1e-12)
check("declination = -obliquity at Dec solstice",
      abs(mk.declination(3 * math.pi / 2.0) + mk.OBLIQUITY_NOW) < 1e-12)
check("declination = 0 at equinox", abs(mk.declination(0.0)) < 1e-12)

# Midnight sun: at the pole in summer the sun never sets (H0 = pi).
check("polar day gives H0 = pi",
      abs(mk.sunrise_hour_angle(math.radians(89.0), mk.declination(math.pi / 2.0)) - math.pi) < 1e-9)
# Polar night: at the pole in winter the sun never rises (H0 = 0), zero insolation.
check("polar night gives zero insolation",
      mk.daily_insolation(math.radians(89.0), 3 * math.pi / 2.0, 0.0167) == 0.0)

# Distance factor: closer than mean at perihelion (nu=0) -> >1, farther at aphelion -> <1.
check("distance factor > 1 at perihelion", mk.distance_factor(0.0, 0.05) > 1.0)
check("distance factor < 1 at aphelion", mk.distance_factor(math.pi, 0.05) < 1.0)
check("distance factor = 1 for circular orbit anywhere",
      abs(mk.distance_factor(1.2, 0.0) - 1.0) < 1e-12)

# Higher obliquity strengthens summer insolation at high latitude.
q_low = mk.summer_solstice_insolation(PHI65, 0.0167, obliquity=math.radians(22.1))
q_high = mk.summer_solstice_insolation(PHI65, 0.0167, obliquity=math.radians(24.5))
check("higher obliquity -> stronger 65N summer sun", q_high > q_low)

# Climatic precession e sin(omega): zero when perihelion at equinox, +e at omega=90 deg.
check("climatic precession = 0 at omega=0", abs(mk.climatic_precession(0.05, 0.0)) < 1e-12)
check("climatic precession = e at omega=90 deg",
      abs(mk.climatic_precession(0.05, math.pi / 2.0) - 0.05) < 1e-12)

# Precession effect: NH summer at perihelion (omega=90) boosts summer sun vs at aphelion.
q_peri = mk.summer_solstice_insolation(PHI65, 0.05, longitude_perihelion=math.pi / 2.0)
q_ap = mk.summer_solstice_insolation(PHI65, 0.05, longitude_perihelion=3 * math.pi / 2.0)
check("summer-at-perihelion beats summer-at-aphelion", q_peri > q_ap)

# Annual-mean insolation nearly independent of obliquity at the equator vs strong at pole:
# equator gets more than pole in the annual mean.
def annual_mean(phi, e=0.0167, eps=mk.OBLIQUITY_NOW):
    n = 360
    return sum(mk.daily_insolation(phi, 2 * math.pi * k / n, e, eps)
               for k in range(n)) / n

check("equator annual-mean insolation exceeds pole",
      annual_mean(0.0) > annual_mean(math.radians(85.0)))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all milankovitch tests passed")

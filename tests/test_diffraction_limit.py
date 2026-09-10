"""Tests for diffraction_limit: Rayleigh resolution, Abbe limit, gratings."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import diffraction_limit as dl

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Hubble: D = 2.4 m, lambda = 550 nm -> theta ~ 0.058 arcsec.
theta_hst = dl.rayleigh_angle_arcsec(550e-9, 2.4)
check("Hubble resolution ~0.05 arcsec", 0.04 < theta_hst < 0.07)

# Rayleigh angle = 1.22 lambda / D.
check("theta = 1.22 lambda / D", abs(dl.rayleigh_angle(500e-9, 0.1) - 1.22 * 500e-9 / 0.1) < 1e-15)

# Bigger aperture -> finer resolution.
check("bigger aperture, smaller angle",
      dl.rayleigh_angle(500e-9, 1.0) < dl.rayleigh_angle(500e-9, 0.1))
# Longer wavelength -> coarser resolution.
check("longer wavelength, larger angle",
      dl.rayleigh_angle(1e-6, 0.1) > dl.rayleigh_angle(500e-9, 0.1))

# Human eye: pupil ~2 mm, 550 nm -> ~1 arcminute (60 arcsec).
theta_eye = dl.rayleigh_angle_arcsec(550e-9, 2e-3)
check("human eye ~1 arcminute", 40.0 < theta_eye < 90.0)

# Linear resolution: theta * distance.
check("linear resolution = theta * distance",
      abs(dl.linear_resolution(500e-9, 0.1, 1000.0) - 1000.0 * dl.rayleigh_angle(500e-9, 0.1)) < 1e-9)

# Abbe limit: visible light, NA ~ 1.4 -> ~200 nm.
d_abbe = dl.abbe_limit(550e-9, 1.4)
check("light microscope Abbe limit ~200 nm", 150e-9 < d_abbe < 250e-9)
# Electron microscope: picometre wavelength -> atomic resolution.
check("electron microscope resolves atoms (< 0.1 nm)",
      dl.abbe_limit(4e-12, 0.02) < 1e-10)
check("higher NA, finer Abbe limit",
      dl.abbe_limit(550e-9, 1.4) < dl.abbe_limit(550e-9, 0.5))

# aperture_for_resolution inverts rayleigh_angle.
th = dl.rayleigh_angle(500e-9, 0.3)
check("aperture_for_resolution inverts", abs(dl.aperture_for_resolution(500e-9, th) - 0.3) < 1e-9)

# Grating: sin(theta) = m lambda / g. 600 lines/mm grating, 500 nm, order 1.
g = 1.0 / 600e3     # line spacing (m) for 600 lines/mm
th1 = dl.grating_angle(500e-9, g, 1)
check("grating first order sin = lambda/g", abs(math.sin(th1) - 500e-9 / g) < 1e-12)
# Higher order -> larger angle.
check("grating order 2 larger angle", dl.grating_angle(500e-9, g, 2) > th1)
# Nonexistent order raises.
try:
    dl.grating_angle(500e-9, g, 5)   # 5*500nm = 2500nm > g ~ 1667nm
    check("nonexistent grating order raises", False)
except ValueError:
    check("nonexistent grating order raises", True)

# Resolving power R = m N.
check("resolving power = m N", dl.grating_resolving_power(2, 10000) == 20000.0)
check("more lines, higher resolving power",
      dl.grating_resolving_power(1, 20000) > dl.grating_resolving_power(1, 5000))
# A 10000-line grating in first order resolves the sodium doublet (589.0 vs 589.6 nm, R~1000).
R = dl.grating_resolving_power(1, 10000)
check("10000-line grating resolves Na doublet", R > 589.3 / 0.6)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all diffraction_limit tests passed")

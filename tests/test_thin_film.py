"""Tests for thin_film: interference colours, AR coatings, Newton's rings."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import thin_film as tf

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Optical path 2 n t.
check("optical path = 2 n t", abs(tf.optical_path(1.33, 300e-9) - 2 * 1.33 * 300e-9) < 1e-18)

# Soap film (n=1.33), 100 nm thick: first-order constructive wavelength.
# 2 n t = (1 - 0.5) lambda -> lambda = 2*1.33*100e-9 / 0.5 = 532 nm (green).
lam_c = tf.constructive_wavelength(1.33, 100e-9, 1, half_wave_shift=True)
check("100 nm soap film bright at ~532 nm", abs(lam_c - 532e-9) < 5e-9)

# Destructive: 2 n t = m lambda -> lambda = 266 nm (m=1, UV) / 133 (m=2)... first visible dark.
lam_d = tf.destructive_wavelength(1.33, 200e-9, 1, half_wave_shift=True)
check("destructive lambda = 2nt/m", abs(lam_d - 2 * 1.33 * 200e-9) < 1e-15)

# Very thin film -> black: both constructive orders push to short wavelengths, and the
# destructive m=1 wavelength for tiny t is huge (destructive across the visible).
t_tiny = 20e-9
# At tiny thickness the round-trip path is far below any visible wavelength, so the
# reflected light is destructive (the classic black soap film before bursting).
check("very thin film optical path << visible", tf.optical_path(1.33, t_tiny) < 100e-9)

# Constructive and destructive interleave: for the same film the constructive wavelength
# (with half-wave shift) is larger than the destructive of the same order.
check("bright wavelength > dark wavelength (same order)",
      tf.constructive_wavelength(1.33, 200e-9, 1) > tf.destructive_wavelength(1.33, 200e-9, 1))

# Anti-reflection quarter-wave thickness: t = lambda/(4n). MgF2 (n=1.38) for 550 nm -> ~100 nm.
t_ar = tf.antireflection_thickness(550e-9, 1.38)
check("MgF2 AR coating ~100 nm", 95e-9 < t_ar < 105e-9)
check("AR thickness = lambda/(4n)", abs(t_ar - 550e-9 / (4 * 1.38)) < 1e-15)
# A quarter-wave layer's round-trip path is half a wavelength (the cancellation condition).
check("quarter-wave round trip = lambda/2",
      abs(tf.optical_path(1.38, t_ar) - 550e-9 / 2.0) < 1e-15)

# Ideal AR index: sqrt(n_out n_sub). Glass (1.52) in air -> ~1.23.
check("ideal AR index for glass ~1.23", abs(tf.ideal_ar_index(1.52) - 1.233) < 0.01)
check("ideal AR index = sqrt(n_out n_sub)", abs(tf.ideal_ar_index(1.52, 1.0) - math.sqrt(1.52)) < 1e-12)
# MgF2 (1.38) is close to but above the ideal for glass -- a practical compromise.
check("MgF2 above ideal glass AR index", 1.38 > tf.ideal_ar_index(1.52))

# Newton's rings: dark ring radius = sqrt(m lambda R), grows as sqrt(m).
R = 1.0   # lens curvature radius (m)
r1 = tf.newton_ring_radius(1, 550e-9, R)
r4 = tf.newton_ring_radius(4, 550e-9, R)
check("Newton dark ring r = sqrt(m lambda R)", abs(r1 - math.sqrt(1 * 550e-9 * R)) < 1e-12)
check("Newton ring 4 twice ring 1 (sqrt m)", abs(r4 - 2.0 * r1) < 1e-12)
# Bright rings sit between dark ones.
check("bright ring between dark rings",
      tf.newton_ring_radius(1, 550e-9, R) < tf.newton_ring_radius(1, 550e-9, R, bright=False) + 1e-9
      and tf.newton_ring_radius(1, 550e-9, R, bright=True) < tf.newton_ring_radius(1, 550e-9, R))
# Larger wavelength -> larger rings.
check("longer wavelength, larger rings",
      tf.newton_ring_radius(1, 700e-9, R) > tf.newton_ring_radius(1, 400e-9, R))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all thin_film tests passed")

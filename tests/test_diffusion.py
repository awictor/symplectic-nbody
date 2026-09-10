"""Tests for diffusion: Fick's laws, Gaussian/erfc profiles, diffusion length."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import diffusion as dz

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


D = 1e-9   # typical small-molecule diffusivity in water (m^2/s)

# Fick's first law: flux runs downhill (negative for positive gradient).
check("flux downhill (J<0 for dC>0)", dz.flux(D, 2.0, 0.1) < 0.0)
check("flux = -D dC/dx", abs(dz.flux(D, 2.0, 0.1) - (-D * 2.0 / 0.1)) < 1e-30)

# rms spread grows as sqrt(t): quadruple time -> double spread.
s1 = dz.rms_spread(100.0, D)
s4 = dz.rms_spread(400.0, D)
check("rms spread doubles when time quadruples", abs(s4 - 2.0 * s1) < 1e-15)
check("rms spread = sqrt(2 D t)", abs(s1 - math.sqrt(2.0 * D * 100.0)) < 1e-18)

# Gaussian profile: peak at origin, symmetric, falls off.
c_peak = dz.gaussian_profile(0.0, 100.0, D)
c_off = dz.gaussian_profile(1e-3, 100.0, D)
check("Gaussian peaks at origin", c_peak > c_off)
check("Gaussian symmetric", abs(dz.gaussian_profile(1e-3, 100.0, D)
                                - dz.gaussian_profile(-1e-3, 100.0, D)) < 1e-30)

# Point release conserves the total amount: integral of C dx = amount.
def integrate_gaussian(t, amount=1.0, n=4001, span=None):
    sigma = dz.rms_spread(t, D)
    span = span or 12.0 * sigma
    dx = span / (n - 1)
    total = 0.0
    for k in range(n):
        x = -span / 2.0 + k * dx
        total += dz.gaussian_profile(x, t, D, amount) * dx
    return total

check("point release conserves amount (~1.0)", abs(integrate_gaussian(50.0) - 1.0) < 1e-3)
check("conserved amount scales with release",
      abs(integrate_gaussian(50.0, amount=5.0) - 5.0) < 5e-3)

# Step (erfc) profile: holds c0/2 at the interface for all time, c0 far behind, 0 far ahead.
check("step profile = c0/2 at interface", abs(dz.step_profile(0.0, 10.0, D, 1.0) - 0.5) < 1e-12)
check("step -> c0 deep on the full side", dz.step_profile(-1.0, 10.0, D, 1.0) > 0.999)
check("step -> 0 deep on the empty side", dz.step_profile(1.0, 10.0, D, 1.0) < 1e-6)
check("step is monotonic decreasing",
      dz.step_profile(-1e-4, 10.0, D) > dz.step_profile(1e-4, 10.0, D))

# Diffusion length and time are inverse: t(L(t)) round-trips.
L = dz.diffusion_length(100.0, D)
check("diffusion_time inverts diffusion_length", abs(dz.diffusion_time(L, D) - 100.0) < 1e-9)
# Quadratic scaling: cross 2x distance takes 4x time.
check("crossing 2x length takes 4x time",
      abs(dz.diffusion_time(2e-4, D) - 4.0 * dz.diffusion_time(1e-4, D)) < 1e-9)

# Small molecule in water: crossing a 10 um cell takes a fraction of a second; a 1 m room
# would take ~30 years -- the reason life is small and rooms need convection.
t_cell = dz.diffusion_time(10e-6, D)
t_room = dz.diffusion_time(1.0, D)
check("10 um cell crossing < 1 s", t_cell < 1.0)
check("1 m room crossing > 1e6 s", t_room > 1e6)

# Stokes-Einstein: ~1 nm sphere in water (eta ~ 1e-3 Pa s) at 298 K ~ 2e-10 .. 3e-9 m^2/s.
Dse = dz.stokes_einstein(298.0, 1e-3, 1e-9)
check("Stokes-Einstein D ~ 2e-10 m^2/s for 1 nm sphere", 1e-10 < Dse < 5e-10)
check("bigger sphere diffuses slower",
      dz.stokes_einstein(298.0, 1e-3, 2e-9) < dz.stokes_einstein(298.0, 1e-3, 1e-9))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all diffusion tests passed")

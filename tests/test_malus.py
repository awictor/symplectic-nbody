"""Tests for malus: polarizer transmission and the cos^2 law."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import malus as ml

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Malus: full transmission at 0, half at 45, zero at 90.
check("full at 0 deg", abs(ml.malus_transmission(1.0, 0.0) - 1.0) < 1e-12)
check("half at 45 deg", abs(ml.malus_transmission(1.0, math.radians(45.0)) - 0.5) < 1e-12)
check("zero at 90 deg", abs(ml.malus_transmission(1.0, math.radians(90.0))) < 1e-12)
check("quarter at 60 deg", abs(ml.malus_transmission(1.0, math.radians(60.0)) - 0.25) < 1e-12)

# Unpolarized through one polarizer: exactly half.
check("unpolarized loses half", abs(ml.unpolarized_through_one(1.0) - 0.5) < 1e-12)

# Two polarizers: (I0/2) cos^2. Crossed -> zero, parallel -> half.
check("crossed polarizers pass nothing",
      abs(ml.two_polarizer_transmission(1.0, math.radians(90.0))) < 1e-12)
check("parallel polarizers pass half",
      abs(ml.two_polarizer_transmission(1.0, 0.0) - 0.5) < 1e-12)
check("two at 45 deg pass quarter",
      abs(ml.two_polarizer_transmission(1.0, math.radians(45.0)) - 0.25) < 1e-12)

# Three-polarizer rescue: crossed pair passes nothing, but a 45-deg middle passes I0/8.
check("three-polarizer 45-deg middle passes I0/8",
      abs(ml.three_polarizer_rescue(1.0, math.radians(45.0)) - 0.125) < 1e-12)
# The rescue is nonzero for any middle angle strictly between 0 and 90.
check("rescue nonzero at 30 deg", ml.three_polarizer_rescue(1.0, math.radians(30.0)) > 0.0)
# And it vanishes when the middle aligns with an outer polarizer (0 or 90).
check("rescue zero when middle at 0", abs(ml.three_polarizer_rescue(1.0, 0.0)) < 1e-12)

# Stack: many small rotations pass nearly all the light (quantum-Zeno-like).
few = ml.stack_transmission(1.0, math.radians(90.0), 2)     # two 45-deg steps -> 0.25
many = ml.stack_transmission(1.0, math.radians(90.0), 100)  # 100 tiny steps -> near 1
check("2-step 90-deg stack passes 1/4", abs(few - 0.25) < 1e-9)
check("100-step 90-deg stack passes most of the light", many > 0.95)
check("more steps, more throughput",
      ml.stack_transmission(1.0, math.radians(90.0), 50) > ml.stack_transmission(1.0, math.radians(90.0), 10))

# Extinction ratio: perfect crossing leaks nothing, small misalignment leaks sin^2.
check("perfect crossing zero leak", abs(ml.extinction_ratio(0.0)) < 1e-12)
check("1 deg misalignment leaks ~sin^2(1deg)",
      abs(ml.extinction_ratio(math.radians(1.0)) - math.sin(math.radians(1.0)) ** 2) < 1e-15)
check("more misalignment, more leak",
      ml.extinction_ratio(math.radians(5.0)) > ml.extinction_ratio(math.radians(1.0)))

# Wave-plate retardance: half-wave (pi) and quarter-wave (pi/2) for the right thickness.
# For delta_n=0.01, lambda=550nm: half-wave t = lambda/(2 delta_n) = 27.5 um.
dn, lam = 0.01, 550e-9
t_half = lam / (2 * dn)
check("half-wave plate retardance = pi",
      abs(ml.waveplate_retardance(t_half, dn, lam) - math.pi) < 1e-9)
t_quarter = lam / (4 * dn)
check("quarter-wave plate retardance = pi/2",
      abs(ml.waveplate_retardance(t_quarter, dn, lam) - math.pi / 2.0) < 1e-9)
check("retardance scales with thickness",
      abs(ml.waveplate_retardance(2 * t_quarter, dn, lam) - 2 * ml.waveplate_retardance(t_quarter, dn, lam)) < 1e-9)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all malus tests passed")

"""Tests for langevin_para: classical paramagnetism and Curie's law."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import langevin_para as lp

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Langevin function: L(0)=0, small-x ~ x/3, large-x -> 1.
check("L(0) = 0", abs(lp.langevin(0.0)) < 1e-12)
check("L(x) ~ x/3 small x", abs(lp.langevin(0.03) - 0.03 / 3.0) < 1e-5)
check("L saturates to 1", lp.langevin(100.0) > 0.98)
check("L monotone increasing", lp.langevin(2.0) > lp.langevin(1.0))
check("L in [0,1]", 0.0 < lp.langevin(1.5) < 1.0)
# Known value L(1) = coth(1) - 1 ~ 0.3130.
check("L(1) ~ 0.313", abs(lp.langevin(1.0) - 0.3130) < 0.001)

# Reduced field x = mu B / kT.
x = lp.reduced_field(lp.BOHR_MAGNETON, 1.0, 300.0)
check("x = mu B / kT", abs(x - lp.BOHR_MAGNETON * 1.0 / (lp.K_B * 300.0)) < 1e-12)
# At room T and 1 T, one Bohr magneton is deep in the linear regime (x ~ 0.0022).
check("Bohr magneton weakly aligned at 300 K, 1 T", x < 0.01)

# Magnetization: per-moment, rises with field, falls with temperature.
check("magnetization rises with field",
      lp.magnetization(lp.BOHR_MAGNETON, 5.0, 300.0) > lp.magnetization(lp.BOHR_MAGNETON, 1.0, 300.0))
check("magnetization falls with temperature",
      lp.magnetization(lp.BOHR_MAGNETON, 1.0, 600.0) < lp.magnetization(lp.BOHR_MAGNETON, 1.0, 300.0))

# Saturation fraction is L(x), between 0 and 1.
sf = lp.saturation_fraction(lp.BOHR_MAGNETON, 1.0, 300.0)
check("saturation fraction small in weak field", sf < 0.01)
# A big moment in a huge field at low T saturates.
check("saturates at high field / low T",
      lp.saturation_fraction(10 * lp.BOHR_MAGNETON, 50.0, 1.0) > 0.98)

# Curie susceptibility ~ 1/T.
chi300 = lp.curie_susceptibility(lp.BOHR_MAGNETON, 300.0)
chi150 = lp.curie_susceptibility(lp.BOHR_MAGNETON, 150.0)
check("Curie chi ~ 1/T (halving T doubles chi)", abs(chi150 - 2 * chi300) < 1e-3 * chi150)
check("chi = n mu^2/(3 kT)",
      abs(chi300 - lp.BOHR_MAGNETON ** 2 / (3 * lp.K_B * 300.0)) < 1e-40)
# Curie constant: chi = C/T.
C = lp.curie_constant(lp.BOHR_MAGNETON)
check("chi = C / T", abs(chi300 - C / 300.0) < 1e-40)

# Small-field magnetization matches chi * B (linear response).
B_small = 0.01
m_lin = lp.magnetization(lp.BOHR_MAGNETON, B_small, 300.0)
check("small-field m ~ chi B", abs(m_lin - chi300 * B_small) < 1e-3 * abs(m_lin))

# field_for_saturation inverts: the field it returns gives the target fraction.
Bf = lp.field_for_saturation(lp.BOHR_MAGNETON, 1.0, 0.5)
check("field_for_saturation hits target",
      abs(lp.saturation_fraction(lp.BOHR_MAGNETON, Bf, 1.0) - 0.5) < 1e-3)
# Higher target fraction needs a bigger field.
check("more alignment needs more field",
      lp.field_for_saturation(lp.BOHR_MAGNETON, 1.0, 0.9) > lp.field_for_saturation(lp.BOHR_MAGNETON, 1.0, 0.5))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all langevin_para tests passed")

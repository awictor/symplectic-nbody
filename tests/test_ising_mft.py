"""Tests for ising_mft: mean-field ferromagnetism and the Curie transition."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ising_mft as im

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Curie temperature = z J.
check("T_c = z J", abs(im.curie_temperature(1.0, 6) - 6.0) < 1e-12)
check("more neighbours, higher T_c", im.curie_temperature(1.0, 8) > im.curie_temperature(1.0, 6))

J, z = 1.0, 4
Tc = im.curie_temperature(J, z)   # = 4

# Spontaneous magnetization: ~0 above T_c, rising toward 1 below.
check("m = 0 above T_c (paramagnet)", im.spontaneous_magnetization(1.5 * Tc, J, z) < 1e-3)
check("m > 0 below T_c (ferromagnet)", im.spontaneous_magnetization(0.5 * Tc, J, z) > 0.5)
check("m -> 1 as T -> 0", im.spontaneous_magnetization(0.01 * Tc, J, z) > 0.99)
check("m rises as T falls",
      im.spontaneous_magnetization(0.3 * Tc, J, z) > im.spontaneous_magnetization(0.9 * Tc, J, z))
check("m in [0,1]", 0.0 <= im.spontaneous_magnetization(0.7 * Tc, J, z) <= 1.0)

# A field induces magnetization even above T_c (paramagnetic response).
check("field magnetizes above T_c",
      im.spontaneous_magnetization(1.5 * Tc, J, z, field=0.5) > 0.0)

# Ferromagnetic classification.
check("below T_c is ferromagnetic", im.is_ferromagnetic(0.5 * Tc, J, z))
check("above T_c is paramagnetic", not im.is_ferromagnetic(1.5 * Tc, J, z))

# Critical scaling m ~ (1 - T/Tc)^(1/2): zero at Tc, beta=1/2 exponent.
check("critical m = 0 at T_c", im.critical_magnetization(Tc, Tc) == 0.0)
check("critical m = 0 above T_c", im.critical_magnetization(1.2 * Tc, Tc) == 0.0)
# At T = 0.75 Tc: m = sqrt(0.25) = 0.5.
check("critical m = 0.5 at 0.75 T_c", abs(im.critical_magnetization(0.75 * Tc, Tc) - 0.5) < 1e-9)
# beta = 1/2: quadrupling the reduced distance doubles m.
m1 = im.critical_magnetization(0.99 * Tc, Tc)
m4 = im.critical_magnetization(0.96 * Tc, Tc)
check("beta=1/2 exponent (4x distance -> 2x m)", abs(m4 - 2 * m1) < 1e-6)

# Curie-Weiss susceptibility diverges at T_c, falls off above.
check("susceptibility diverges at T_c", math.isinf(im.curie_weiss_susceptibility(Tc, Tc)))
check("susceptibility positive above T_c", im.curie_weiss_susceptibility(2 * Tc, Tc) > 0.0)
check("susceptibility falls with temperature",
      im.curie_weiss_susceptibility(3 * Tc, Tc) < im.curie_weiss_susceptibility(1.5 * Tc, Tc))
# chi = C/(T-Tc).
check("chi = C/(T-Tc)", abs(im.curie_weiss_susceptibility(2 * Tc, Tc, 2.0) - 2.0 / Tc) < 1e-9)

# Reduced temperature: negative below, zero at, positive above T_c.
check("reduced t < 0 below T_c", im.reduced_temperature(0.5 * Tc, Tc) < 0)
check("reduced t = 0 at T_c", abs(im.reduced_temperature(Tc, Tc)) < 1e-12)
check("reduced t > 0 above T_c", im.reduced_temperature(2 * Tc, Tc) > 0)

# Self-consistent m and the sqrt-scaling approx both vanish at T_c and are nonzero just
# below (the closed form omits the sqrt(3) MFT prefactor, so allow a loose band).
m_sc = im.spontaneous_magnetization(0.9 * Tc, J, z)
m_crit = im.critical_magnetization(0.9 * Tc, Tc)
check("both scalings nonzero just below T_c", m_sc > 0.1 and m_crit > 0.1)
check("self-consistent m exceeds bare sqrt approx near T_c", m_sc > m_crit)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all ising_mft tests passed")

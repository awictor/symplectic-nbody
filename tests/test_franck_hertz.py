"""Tests for franck_hertz: quantized electron energy loss."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import franck_hertz as fh

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Excitation voltage equals the energy in eV: mercury 4.9 eV -> 4.9 V.
check("mercury excitation voltage 4.9 V", abs(fh.excitation_voltage(4.9) - 4.9) < 1e-9)

# Dip spacing is the excitation voltage.
check("dip spacing = 4.9 V for mercury", abs(fh.dip_spacing(4.9) - 4.9) < 1e-9)

# Dip voltages: evenly spaced multiples.
dips = fh.dip_voltages(4.9, 4)
check("first dip at 4.9 V", abs(dips[0] - 4.9) < 1e-9)
check("dips evenly spaced by 4.9 V",
      all(abs((dips[i + 1] - dips[i]) - 4.9) < 1e-9 for i in range(len(dips) - 1)))
check("four dips returned", len(dips) == 4)
# Fourth dip at ~19.6 V.
check("fourth dip at ~19.6 V", abs(dips[3] - 19.6) < 1e-9)

# Offset (contact potential) shifts all dips.
dips_off = fh.dip_voltages(4.9, 3, offset=1.5)
check("offset shifts first dip", abs(dips_off[0] - (1.5 + 4.9)) < 1e-9)
check("offset preserves spacing", abs((dips_off[1] - dips_off[0]) - 4.9) < 1e-9)

# Number of excitations: 15 V / 4.9 -> 3.
check("15 V drives 3 excitations", fh.num_excitations(15.0, 4.9) == 3)
check("4 V drives 0 excitations (below threshold)", fh.num_excitations(4.0, 4.9) == 0)
check("exactly 9.8 V drives 2", fh.num_excitations(9.8, 4.9) == 2)
check("more voltage, more excitations",
      fh.num_excitations(30.0, 4.9) > fh.num_excitations(15.0, 4.9))

# Residual energy: V - n dV, always in [0, dV).
res = fh.residual_energy(15.0, 4.9)
check("residual = 15 - 3*4.9 = 0.3 eV", abs(res - (15.0 - 3 * 4.9)) < 1e-9)
check("residual below one quantum", 0 <= fh.residual_energy(23.0, 4.9) < 4.9)
# Just below a dip, residual is nearly a full quantum; just above, nearly zero.
check("residual near zero just past a dip", fh.residual_energy(9.85, 4.9) < 0.1)

# Emission wavelength: mercury 4.9 eV -> ~253 nm (UV).
lam = fh.emission_wavelength(4.9)
check("mercury emission ~254 nm", 250e-9 < lam < 258e-9)
# Higher excitation energy -> shorter wavelength.
check("higher energy, shorter wavelength",
      fh.emission_wavelength(10.0) < fh.emission_wavelength(4.9))
# Neon (~18.7 eV excitation) emits in the deep UV.
check("neon shorter wavelength than mercury",
      fh.emission_wavelength(18.7) < fh.emission_wavelength(4.9))

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all franck_hertz tests passed")

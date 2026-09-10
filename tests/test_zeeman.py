"""Tests for zeeman: magnetic splitting of spectral lines."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import zeeman as zm

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# Bohr magneton ~9.274e-24 J/T.
check("Bohr magneton ~9.274e-24", abs(zm.bohr_magneton() - 9.274e-24) < 0.01e-24)

# Normal Zeeman shift ~14.0 GHz/T.
check("normal shift ~14 GHz at 1 T", abs(zm.normal_zeeman_shift_hz(1.0) - 14.0e9) < 0.2e9)
# Linear in field.
check("shift linear in field", abs(zm.normal_zeeman_shift_hz(2.0) - 2.0 * zm.normal_zeeman_shift_hz(1.0)) < 1e3)

# Wavelength shift at 500 nm, 1 T: delta_lambda = lambda^2 dnu/c.
dl = zm.normal_zeeman_shift_wavelength(1.0, 500e-9)
check("wavelength shift ~12 pm at 1 T", 5e-12 < dl < 2e-11)
check("wavelength shift = lambda^2 dnu / c",
      abs(dl - 500e-9**2 * zm.normal_zeeman_shift_hz(1.0) / zm.C) < 1e-20)
# Bigger field, bigger split.
check("bigger field, bigger wavelength split",
      zm.normal_zeeman_shift_wavelength(3.0, 500e-9) > dl)

# Lande g-factor: pure orbital (S=0, J=L) -> g=1; pure spin (L=0, J=S) -> g=2.
check("pure orbital g = 1", abs(zm.lande_g(1, 1, 0) - 1.0) < 1e-9)
check("pure spin g = 2", abs(zm.lande_g(0.5, 0, 0.5) - 2.0) < 1e-9)
# Sodium D lines: 2P_1/2 (J=1/2,L=1,S=1/2) g=2/3; 2P_3/2 (J=3/2) g=4/3; 2S_1/2 g=2.
check("Na 2P1/2 g = 2/3", abs(zm.lande_g(0.5, 1, 0.5) - 2.0/3.0) < 1e-9)
check("Na 2P3/2 g = 4/3", abs(zm.lande_g(1.5, 1, 0.5) - 4.0/3.0) < 1e-9)
check("Na 2S1/2 g = 2", abs(zm.lande_g(0.5, 0, 0.5) - 2.0) < 1e-9)
# g=0 for J=0 (no splitting).
check("J=0 gives g=0", zm.lande_g(0, 0, 0) == 0.0)

# Energy shift: g m_J mu_B B, symmetric in m_J.
check("energy shift = g m_J mu_B B",
      abs(zm.energy_shift(2.0, 0.5, 1.0) - 2.0 * 0.5 * zm.bohr_magneton()) < 1e-30)
check("shift flips sign with m_J",
      abs(zm.energy_shift(2.0, -0.5, 1.0) + zm.energy_shift(2.0, 0.5, 1.0)) < 1e-30)
check("m_J = 0 no shift", zm.energy_shift(1.5, 0.0, 1.0) == 0.0)

# Anomalous shift in Hz.
check("anomalous shift = E/h",
      abs(zm.anomalous_shift_hz(2.0, 0.5, 1.0) - zm.energy_shift(2.0, 0.5, 1.0) / zm.H) < 1)

# field_from_splitting inverts normal shift.
B = zm.field_from_splitting(zm.normal_zeeman_shift_hz(0.3))
check("field_from_splitting inverts", abs(B - 0.3) < 1e-9)
# Solar sunspot field ~0.3 T gives a measurable ~4 GHz split.
check("sunspot 0.3 T split ~4 GHz", 3e9 < zm.normal_zeeman_shift_hz(0.3) < 5e9)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all zeeman tests passed")

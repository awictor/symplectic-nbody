"""Shakura-Sunyaev accretion-disk tests.

Claims checked:
  1. The disk temperature falls as r^(-3/4) far from the inner edge and vanishes at
     the inner edge (the boundary factor).
  2. A stellar-mass black-hole disk peaks in soft X-rays (~10^7 K, ~keV); a
     supermassive one peaks in the UV (~10^5 K) -- the quasar big blue bump.
  3. The Eddington luminosity is ~1.3e31 W per solar mass and scales with M.
  4. Accretion luminosity L = eta Mdot c^2 dwarfs fusion's ~0.007 efficiency.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accretion_disk import (schwarzschild_radius, isco_radius,  # noqa: E402
                            disk_temperature, characteristic_temperature,
                            peak_temperature, eddington_luminosity,
                            radiative_efficiency, luminosity, M_SUN, C,
                            ETA_SCHWARZSCHILD)

K_B = 1.380649e-23
KEV = 1.602176634e-16

M10 = 10 * M_SUN
MDOT10 = eddington_luminosity(M10) / (ETA_SCHWARZSCHILD * C * C)  # Eddington rate


def test_isco_is_three_schwarzschild_radii():
    assert abs(isco_radius(M10) / schwarzschild_radius(M10) - 3.0) < 1e-9


def test_temperature_r_minus_three_quarters():
    r_in = isco_radius(M10)
    # far from the edge the boundary factor -> 1 and T ~ r^(-3/4)
    t1 = disk_temperature(100 * r_in, M10, MDOT10)
    t2 = disk_temperature(400 * r_in, M10, MDOT10)
    assert abs(t2 / t1 - 4.0 ** (-0.75)) < 0.02


def test_temperature_vanishes_at_inner_edge():
    r_in = isco_radius(M10)
    assert disk_temperature(r_in, M10, MDOT10) == 0.0


def test_stellar_disk_is_xray_hot():
    T = peak_temperature(M10, MDOT10)
    assert 3e6 < T < 3e7, f"stellar-mass disk peak T {T} K not soft-X-ray"
    kT_keV = K_B * T / KEV
    assert 0.2 < kT_keV < 3.0, f"peak kT {kT_keV} keV not ~1 keV"


def test_supermassive_disk_is_uv():
    Mb = 1e8 * M_SUN
    mdot = eddington_luminosity(Mb) / (ETA_SCHWARZSCHILD * C * C)
    T = peak_temperature(Mb, mdot)
    assert 3e4 < T < 5e5, f"SMBH disk peak T {T} K not UV"


def test_eddington_scales_with_mass():
    assert abs(eddington_luminosity(2 * M_SUN) / eddington_luminosity(M_SUN) - 2.0) < 1e-9
    # ~1.26e31 W per solar mass
    assert abs(eddington_luminosity(M_SUN) / 1.26e31 - 1.0) < 0.05


def test_accretion_beats_fusion():
    # black-hole accretion efficiency (~0.057) is ~8x hydrogen fusion (~0.007)
    assert radiative_efficiency() > 0.05
    assert radiative_efficiency() / 0.007 > 5.0


def test_luminosity_formula():
    L = luminosity(MDOT10)
    assert abs(L - ETA_SCHWARZSCHILD * MDOT10 * C * C) < 1.0
    # accreting at the Eddington rate returns the Eddington luminosity
    assert abs(L / eddington_luminosity(M10) - 1.0) < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

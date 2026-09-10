"""CMB acoustic-scale tests.

Claims checked:
  1. The first acoustic peak lands at multipole l ~ 220 (the WMAP/Planck value),
     i.e. features about 1 degree across.
  2. The comoving distance to the last-scattering surface is ~14000 Mpc.
  3. The acoustic angle is sub-degree, and l = pi/theta.
  4. The photon-baryon sound speed is below c/sqrt(3) and drops as baryon loading
     R rises.
  5. A flatter (more matter, less dark energy) universe shifts the peak, so l
     depends on cosmology -- which is how the CMB measures flatness.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from friedmann import Cosmology  # noqa: E402
from saha import recombination_redshift  # noqa: E402
from cmb import (sound_speed_fraction, sound_horizon, acoustic_angle,  # noqa: E402
                 first_peak_multipole, angular_diameter_distance_to_rec)
from distances import comoving_distance  # noqa: E402

LCDM = Cosmology(Omega_m=0.3, Omega_L=0.7)
Z_REC = recombination_redshift(0.5)


def test_first_peak_multipole():
    l = first_peak_multipole(LCDM, Z_REC)
    assert 180 < l < 260, f"first peak l {l} not ~220"


def test_distance_to_last_scattering():
    D_C = comoving_distance(LCDM, Z_REC) * 299792.458 / 70.0  # Mpc
    assert 12000 < D_C < 16000, f"comoving distance to LSS {D_C} Mpc not ~14000"


def test_acoustic_angle_and_multipole_consistency():
    theta = acoustic_angle(LCDM, Z_REC)
    assert math.degrees(theta) < 1.5, "acoustic angle should be sub-degree-ish"
    assert abs(first_peak_multipole(LCDM, Z_REC) - math.pi / theta) < 1e-6


def test_sound_speed_limits():
    assert abs(sound_speed_fraction(0.0) - 1.0 / math.sqrt(3.0)) < 1e-12
    assert sound_speed_fraction(1.0) < sound_speed_fraction(0.0), "baryons slow the sound"


def test_cosmology_dependence():
    # a different matter fraction moves the peak -> the CMB constrains cosmology
    open_univ = Cosmology(Omega_m=0.3, Omega_L=0.0)  # Omega_k != 0
    l_lcdm = first_peak_multipole(LCDM, Z_REC)
    l_open = first_peak_multipole(open_univ, Z_REC)
    assert abs(l_lcdm - l_open) > 1.0, "peak multipole should depend on geometry"


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

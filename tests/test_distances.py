"""Cosmological-distance tests.

Claims checked:
  1. At low redshift the luminosity distance obeys the Hubble law D_L ~ (c/H0) z.
  2. A dark-energy universe puts a given redshift at a LARGER luminosity
     distance than a decelerating (Einstein-de Sitter) universe, so type-Ia
     supernovae look fainter -- the evidence for cosmic acceleration.
  3. The angular-diameter distance is non-monotonic, peaking near z ~ 1.6 for
     LCDM (why the CMB spots look ~1 degree across).
  4. D_L, D_A, and comoving distance satisfy D_L = (1+z)^2 D_A (flat universe).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from friedmann import Cosmology  # noqa: E402
from distances import (comoving_distance, luminosity_distance,  # noqa: E402
                       angular_diameter_distance, distance_modulus,
                       angular_diameter_peak)

LCDM = Cosmology(Omega_m=0.3, Omega_L=0.7)
EDS = Cosmology(Omega_m=1.0, Omega_L=0.0)


def test_hubble_law_at_low_z():
    z = 1e-3
    assert abs(luminosity_distance(LCDM, z) / z - 1.0) < 1e-2, "low-z D_L should be ~z"


def test_dark_energy_makes_sne_fainter():
    for z in (0.5, 1.0):
        assert luminosity_distance(LCDM, z) > luminosity_distance(EDS, z), \
            f"LCDM D_L should exceed EdS at z={z}"
        assert distance_modulus(LCDM, z) > distance_modulus(EDS, z), \
            f"LCDM SNe should be fainter (larger mu) at z={z}"


def test_angular_diameter_turnover():
    z_peak = angular_diameter_peak(LCDM, z_max=5.0, n=500)
    assert 1.3 < z_peak < 1.9, f"D_A peak {z_peak} not near 1.6"
    # D_A actually decreases past the peak
    assert angular_diameter_distance(LCDM, 4.0) < angular_diameter_distance(LCDM, z_peak)


def test_distance_duality():
    # Etherington relation for a flat universe: D_L = (1+z)^2 D_A
    for z in (0.3, 1.0, 2.0):
        dl = luminosity_distance(LCDM, z)
        da = angular_diameter_distance(LCDM, z)
        assert abs(dl - (1 + z) ** 2 * da) < 1e-9, f"duality fails at z={z}"


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

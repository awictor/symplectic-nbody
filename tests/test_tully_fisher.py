"""Tully-Fisher relation tests.

Claims checked:
  1. Luminosity scales as v_flat^4: doubling the rotation speed brightens a spiral
     16-fold.
  2. A Milky-Way-like spiral (v ~ 220 km/s) has L ~ few x 10^10 L_sun and a baryonic
     mass ~ 10^11 M_sun.
  3. Faster rotators are intrinsically brighter (more negative absolute magnitude).
  4. The relation inverts: rotation speed <-> luminosity, and gives a distance from
     an apparent magnitude.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tully_fisher import (luminosity_from_vflat, absolute_magnitude,  # noqa: E402
                          distance_from_apparent, vflat_from_luminosity,
                          baryonic_mass_from_vflat)


def test_l_v_fourth():
    assert abs(luminosity_from_vflat(400) / luminosity_from_vflat(200) - 16.0) < 1e-6


def test_milky_way_luminosity():
    L = luminosity_from_vflat(220)
    assert 1e10 < L < 6e10, f"MW-like luminosity {L} L_sun off scale"


def test_milky_way_baryonic_mass():
    M = baryonic_mass_from_vflat(220)
    assert 5e10 < M < 3e11, f"MW baryonic mass {M} M_sun off scale"


def test_faster_brighter():
    assert absolute_magnitude(300) < absolute_magnitude(150)


def test_luminosity_inverts():
    for v in (100.0, 180.0, 300.0):
        L = luminosity_from_vflat(v)
        assert abs(vflat_from_luminosity(L) - v) < 1e-6


def test_baryonic_v_fourth():
    assert abs(baryonic_mass_from_vflat(400) / baryonic_mass_from_vflat(200) - 16.0) < 1e-6


def test_distance_positive_and_farther_fainter():
    d1 = distance_from_apparent(180.0, 12.0)
    d2 = distance_from_apparent(180.0, 14.0)   # same galaxy type, fainter -> farther
    assert d2 > d1 > 0


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

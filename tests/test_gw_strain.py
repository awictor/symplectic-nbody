"""Gravitational-wave-strain tests (GW150914).

Claims checked:
  1. The chirp mass of the GW150914 binary (36 + 29 solar masses) is ~28-30
     solar masses.
  2. Its strain amplitude at Earth is ~1e-21 (within a factor of a few of the
     measured value, from the leading-order amplitude formula).
  3. The LIGO arm-length change is ~1e-18 m -- a fraction of a proton width.
  4. Strain scales as 1/distance and as f^{2/3}.
  5. The chirp mass is symmetric in the two masses and rises with total mass.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gw_strain import (chirp_mass, strain, arm_length_change,  # noqa: E402
                       gw150914_strain, M_SUN, MPC)


def test_gw150914_chirp_mass():
    Mc = chirp_mass(36 * M_SUN, 29 * M_SUN) / M_SUN
    assert 26.0 < Mc < 32.0, f"GW150914 chirp mass {Mc} not ~28-30 M_sun"


def test_gw150914_strain_order():
    h = gw150914_strain()
    assert 3e-22 < h < 5e-21, f"GW150914 strain {h} not ~1e-21"


def test_arm_length_change_subproton():
    dL = arm_length_change(gw150914_strain())
    assert 1e-18 < dL < 3e-17, f"arm change {dL} m off"
    assert dL < 8e-16, "should be a fraction of a proton width"


def test_strain_inverse_distance():
    h1 = strain(36 * M_SUN, 29 * M_SUN, 410 * MPC, 150.0)
    h2 = strain(36 * M_SUN, 29 * M_SUN, 820 * MPC, 150.0)
    assert abs(h1 / h2 - 2.0) < 1e-9, "strain should scale as 1/distance"


def test_strain_frequency_scaling():
    h1 = strain(36 * M_SUN, 29 * M_SUN, 410 * MPC, 150.0)
    h2 = strain(36 * M_SUN, 29 * M_SUN, 410 * MPC, 300.0)
    assert abs(h2 / h1 - 2 ** (2.0 / 3.0)) < 1e-9, "strain should scale as f^{2/3}"


def test_chirp_mass_symmetry_and_monotonicity():
    assert abs(chirp_mass(30 * M_SUN, 20 * M_SUN) - chirp_mass(20 * M_SUN, 30 * M_SUN)) < 1e10
    assert chirp_mass(60 * M_SUN, 40 * M_SUN) > chirp_mass(30 * M_SUN, 20 * M_SUN)


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

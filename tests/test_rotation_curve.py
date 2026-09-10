"""Galaxy rotation-curve tests: the dark-matter argument, made quantitative.

Claims checked:
  1. A visible-disk-only rotation curve declines Keplerian-ly (outer log-log
     slope ~ -1/2) beyond the disk, because the enclosed mass levels off.
  2. Adding an NFW dark halo flattens the outer curve (slope ~ 0), matching what
     is actually observed -- the evidence for dark matter.
  3. The NFW enclosed mass keeps growing ~ r at large radius (that is why the
     curve stays flat), while the disk mass saturates.
  4. Circular speed obeys v_c = sqrt(G M(<r)/r).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotation_curve import (rotation_curve, keplerian_tail_slope,  # noqa: E402
                            disk_enclosed_mass, nfw_enclosed_mass, circular_speed)

RADII = [0.1 + 0.2 * i for i in range(60)]


def test_visible_only_declines_keplerian():
    v_vis, _ = rotation_curve(RADII, M_disk=1.0, R_d=1.0, halo=None)
    slope = keplerian_tail_slope(RADII, v_vis)
    assert abs(slope - (-0.5)) < 0.1, f"visible-only outer slope {slope} not ~ -0.5"


def test_halo_flattens_curve():
    _, v_tot = rotation_curve(RADII, M_disk=1.0, R_d=1.0, halo=(0.02, 5.0))
    slope = keplerian_tail_slope(RADII, v_tot)
    assert abs(slope) < 0.1, f"disk+halo outer slope {slope} not ~ flat"


def test_disk_mass_saturates_halo_grows():
    r1, r2 = 8.0, 16.0
    d1 = disk_enclosed_mass(r1, 1.0, 1.0)
    d2 = disk_enclosed_mass(r2, 1.0, 1.0)
    # disk enclosed mass barely changes past many scale lengths (< 1%)
    assert (d2 - d1) / d2 < 0.01, f"disk mass should saturate: {d1} -> {d2}"
    h1 = nfw_enclosed_mass(r1, 0.02, 5.0)
    h2 = nfw_enclosed_mass(r2, 0.02, 5.0)
    # halo enclosed mass keeps climbing substantially
    assert h2 > 1.3 * h1, f"halo mass should keep growing: {h1} -> {h2}"


def test_circular_speed_formula():
    assert abs(circular_speed(4.0, 8.0, G=2.0) - math.sqrt(2.0 * 8.0 / 4.0)) < 1e-12


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

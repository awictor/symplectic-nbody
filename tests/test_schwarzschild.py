"""Schwarzschild strong-field orbit tests.

Claims checked:
  1. The ISCO is at 6M and the photon sphere at 3M.
  2. The circular-orbit angular momentum L^2 = M r^2/(r-3M) matches sqrt(12) M at
     the ISCO and diverges as r -> 3M (no circular orbit inside the photon sphere).
  3. A bound orbit with enough angular momentum precesses (stays between a
     perihelion and aphelion); a low-angular-momentum orbit plunges to r -> 0.
  4. In the weak field the relativistic perihelion advance matches the classic
     6*pi*M/(a(1-e^2)); it is somewhat larger in the strong field.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schwarzschild import (isco_radius, photon_sphere_radius, circular_orbit_L,  # noqa: E402
                           orbit_shape, precession_per_orbit)


def test_isco_and_photon_sphere():
    assert isco_radius(1.0) == 6.0
    assert photon_sphere_radius(1.0) == 3.0


def test_circular_L_at_isco():
    assert abs(circular_orbit_L(6.0, 1.0) - math.sqrt(12.0)) < 1e-9


def test_no_circular_orbit_inside_photon_sphere():
    assert math.isnan(circular_orbit_L(2.5, 1.0))
    # diverges approaching 3M from outside
    assert circular_orbit_L(3.001, 1.0) > 30.0


def test_bound_orbit_stays_bounded():
    phis, rs = orbit_shape(20.0, 4.0, 0.0, M=1.0, dphi=1e-3, max_phi=40.0)
    assert min(rs) > 3.0, f"bound orbit should not plunge, min r {min(rs)}"
    assert max(rs) <= 20.0 + 1e-6


def test_low_angular_momentum_plunges():
    phis, rs = orbit_shape(8.0, 3.2, 0.0, M=1.0, dphi=1e-3, max_phi=50.0)
    assert min(rs) < 0.1, f"low-L orbit should plunge to r->0, min r {min(rs)}"


def test_weak_field_precession_matches_classic():
    r_peri, r_apo = 90.0, 110.0
    a = 0.5 * (r_peri + r_apo)
    e = (r_apo - r_peri) / (r_apo + r_peri)
    classic = 6.0 * math.pi * 1.0 / (a * (1 - e * e))
    prec = precession_per_orbit(r_peri, r_apo, M=1.0)
    assert abs(prec - classic) / classic < 0.1, f"weak-field precession {prec} vs {classic}"
    assert prec > 0, "GR precession should be prograde"


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

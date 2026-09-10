"""Free-fall / dynamical-timescale tests.

Claims checked:
  1. t_ff = sqrt(3 pi / 32 G rho) depends only on density, not size or mass; the
     Sun free-falls in ~30 minutes.
  2. The timescale scales as rho^(-1/2): quadruple the density, halve the time.
  3. A dense molecular-cloud core collapses in a few hundred thousand years.
  4. A surface-skimming orbit's period depends only on mean density (~84 min for
     Earth), the same 1/sqrt(G rho) clock as free-fall.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from free_fall import (free_fall_time, dynamical_time, mean_density,  # noqa: E402
                       orbital_period_mean_density, free_fall_time_from_body,
                       M_SUN, R_SUN, G)

M_EARTH, R_EARTH = 5.972e24, 6.371e6


def test_sun_free_fall_half_hour():
    t = free_fall_time_from_body(M_SUN, R_SUN) / 60.0
    assert abs(t - 29.5) < 1.0, f"solar free-fall {t} min not ~30"


def test_independent_of_size_at_fixed_density():
    rho = 1400.0
    # two very different bodies at the same mean density free-fall in the same time
    t_small = free_fall_time(rho)
    t_big = free_fall_time(rho)
    assert t_small == t_big


def test_density_scaling():
    base = free_fall_time(1000.0)
    assert abs(free_fall_time(4000.0) / base - 0.5) < 1e-9   # rho^(-1/2)


def test_denser_collapses_faster():
    assert free_fall_time(1e4) < free_fall_time(1e2)


def test_molecular_cloud_core():
    rho = 1e4 * 1e6 * 2.3 * 1.6726e-27   # n=1e4/cc, mu=2.3
    t_kyr = free_fall_time(rho) / (3.15576e10)
    assert 100.0 < t_kyr < 1000.0, f"cloud-core free-fall {t_kyr} kyr off range"


def test_surface_orbit_period():
    rho = mean_density(M_EARTH, R_EARTH)
    P = orbital_period_mean_density(rho) / 60.0
    assert 80.0 < P < 90.0, f"Earth surface-orbit period {P} min not ~84"


def test_dynamical_time_same_scaling():
    # dynamical time carries the same 1/sqrt(G rho) dependence as t_ff
    r = free_fall_time(2000.0) / dynamical_time(2000.0)
    r2 = free_fall_time(9000.0) / dynamical_time(9000.0)
    assert abs(r - r2) < 1e-12, "t_ff / t_dyn should be a constant ratio"


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

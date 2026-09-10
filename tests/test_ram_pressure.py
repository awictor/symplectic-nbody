"""Ram-pressure-stripping (Gunn-Gott) tests.

Claims checked:
  1. Ram pressure P = rho v^2 scales with ICM density and with v^2.
  2. A Milky-Way-like disk falling through a rich cluster is stripped in its
     outskirts but keeps its dense inner gas (a finite stripping radius).
  3. A harsher environment (denser ICM, faster infall) strips more deeply
     (smaller surviving radius).
  4. The Gunn-Gott verdict matches comparing P_ram to 2 pi G Sigma_s Sigma_g.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ram_pressure import (ram_pressure, icm_density, restoring_pressure,  # noqa: E402
                          is_stripped, stripping_radius, M_SUN, KPC)

PC = KPC / 1000.0
SIGMA0_STAR = 800.0 * M_SUN / PC ** 2   # central stellar surface density
SIGMA0_GAS = 40.0 * M_SUN / PC ** 2
H = 3.0 * KPC                           # disk scale length


def test_ram_pressure_scalings():
    rho = icm_density(1e-3)
    base = ram_pressure(rho, 1000e3)
    assert abs(ram_pressure(rho, 2000e3) / base - 4.0) < 1e-9   # v^2
    assert abs(ram_pressure(2 * rho, 1000e3) / base - 2.0) < 1e-9  # rho


def test_finite_stripping_radius_in_cluster():
    rho = icm_density(1e-3)
    R = stripping_radius(rho, 1500e3, SIGMA0_STAR, SIGMA0_GAS, H) / KPC
    assert 1.0 < R < 10.0, f"stripping radius {R} kpc not a partial strip"


def test_inner_gas_survives_outer_stripped():
    rho = icm_density(1e-3)
    # centre (high Sigma) holds; outskirts (low Sigma) go
    assert not is_stripped(rho, 1500e3, SIGMA0_STAR, SIGMA0_GAS)
    assert is_stripped(rho, 1500e3, 0.02 * SIGMA0_STAR, 0.02 * SIGMA0_GAS)


def test_harsher_environment_strips_deeper():
    mild = stripping_radius(icm_density(1e-4), 500e3, SIGMA0_STAR, SIGMA0_GAS, H)
    harsh = stripping_radius(icm_density(1e-3), 1500e3, SIGMA0_STAR, SIGMA0_GAS, H)
    assert harsh < mild, "a denser, faster environment should strip to smaller R"


def test_gunn_gott_matches_pressures():
    rho, v = icm_density(1e-3), 1200e3
    verdict = is_stripped(rho, v, SIGMA0_STAR, SIGMA0_GAS)
    direct = ram_pressure(rho, v) > restoring_pressure(SIGMA0_STAR, SIGMA0_GAS)
    assert verdict == direct


def test_everything_stripped_returns_zero():
    # enormous ram pressure strips even the centre -> R = 0
    R = stripping_radius(icm_density(1.0), 3000e3, SIGMA0_STAR, SIGMA0_GAS, H)
    assert R == 0.0


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

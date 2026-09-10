"""Rayleigh-Benard convection tests.

Claims checked:
  1. Convection sets in above the critical Rayleigh number Ra_c ~ 1708 (rigid
     boundaries); the free-free value is (27/4) pi^4 ~ 657.5.
  2. A pot of water heated on a stove is enormously supercritical (Ra ~ 1e7).
  3. The critical temperature difference makes Ra = Ra_c exactly, and below it the
     layer only conducts (Nu = 1).
  4. Above onset the Nusselt number grows as (Ra/Ra_c)^(1/3).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rayleigh_benard import (rayleigh_number, is_convecting,  # noqa: E402
                             critical_delta_T, nusselt_number,
                             thermal_diffusivity, RA_CRITICAL_RIGID,
                             RA_CRITICAL_FREE)

# water properties
ALPHA, NU, KAPPA = 2.6e-4, 1e-6, 1.4e-7


def test_free_free_critical_value():
    assert abs(RA_CRITICAL_FREE - 27.0 / 4.0 * math.pi ** 4) < 0.01


def test_water_pot_convects():
    Ra = rayleigh_number(10.0, 0.05, ALPHA, NU, KAPPA)
    assert Ra > 1e6
    assert is_convecting(Ra)


def test_critical_dt_gives_ra_c():
    d = 0.01
    dTc = critical_delta_T(d, ALPHA, NU, KAPPA)
    Ra = rayleigh_number(dTc, d, ALPHA, NU, KAPPA)
    assert abs(Ra - RA_CRITICAL_RIGID) < 1.0


def test_below_onset_conducts():
    d = 0.01
    dTc = critical_delta_T(d, ALPHA, NU, KAPPA)
    Ra = rayleigh_number(0.5 * dTc, d, ALPHA, NU, KAPPA)
    assert not is_convecting(Ra)
    assert nusselt_number(Ra) == 1.0


def test_nusselt_one_third_law():
    Ra = 8.0 * RA_CRITICAL_RIGID
    # Nu ~ (Ra/Ra_c)^(1/3): 8x critical -> 2x heat transport
    assert abs(nusselt_number(Ra) - 2.0) < 1e-9


def test_deeper_layer_lower_critical_dt():
    shallow = critical_delta_T(0.01, ALPHA, NU, KAPPA)
    deep = critical_delta_T(0.05, ALPHA, NU, KAPPA)
    assert deep < shallow, "deeper layers convect at a smaller temperature drop"


def test_thermal_diffusivity():
    # water: k~0.6, rho~1000, c_p~4180 -> kappa ~ 1.4e-7
    kap = thermal_diffusivity(0.6, 1000.0, 4180.0)
    assert abs(kap - 1.4e-7) < 2e-8


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

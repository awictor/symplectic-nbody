"""Photon-photon pair-production tests.

Claims checked:
  1. The electron rest energy is 511 keV; two 511 keV photons just pair-produce
     head-on, two slightly lower ones do not.
  2. The partner-energy threshold is (m_e c^2)^2 / E_background: a 1 eV background
     photon needs a ~261 GeV gamma ray.
  3. The gamma-ray horizon: TeV gammas are absorbed by eV extragalactic
     background light, PeV gammas by the meV CMB.
  4. A larger collision angle (up to head-on) lowers the energy threshold.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pair_production import (electron_rest_energy_kev, can_pair_produce,  # noqa: E402
                             head_on_threshold, partner_threshold_energy,
                             gamma_ray_horizon_case, KEV, EV, GEV, MEC2)  # noqa: F401


def test_electron_rest_energy():
    assert abs(electron_rest_energy_kev() - 511.0) < 1.0


def test_511_kev_threshold():
    assert can_pair_produce(511 * KEV, 511 * KEV), "two 511 keV should just pair"
    assert not can_pair_produce(500 * KEV, 500 * KEV), "just below threshold"


def test_head_on_threshold():
    assert abs(head_on_threshold() - MEC2) < 1e-30


def test_partner_threshold():
    E = partner_threshold_energy(1 * EV) / GEV
    assert abs(E - 261.0) < 5.0, f"partner threshold for 1 eV {E} GeV not ~261"


def test_gamma_ray_horizon():
    assert gamma_ray_horizon_case(1.0, 1.0), "1 TeV absorbed by 1 eV EBL"
    assert not gamma_ray_horizon_case(1.0, 0.1), "1 TeV survives a 0.1 eV field"
    assert gamma_ray_horizon_case(1000.0, 6e-4), "PeV absorbed by the CMB"


def test_angle_lowers_threshold():
    # just above the head-on threshold, forward collision cannot pair-produce
    # (the (1-cos theta) factor is too small) but head-on can
    E = 520 * KEV
    assert not can_pair_produce(E, E, theta_rad=0.1), "shallow angle below threshold"
    assert can_pair_produce(E, E, theta_rad=math.pi), "head-on above threshold"


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

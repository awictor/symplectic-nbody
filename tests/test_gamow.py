"""Gamow-peak fusion tests.

Claims checked:
  1. Proton-proton fusion in the Sun (T=1.5e7 K) peaks at ~6 keV -- several times the
     mean thermal energy kT ~ 1.3 keV, far out on the Maxwell-Boltzmann tail.
  2. The reaction integrand exp(-E/kT - sqrt(E_G/E)) is maximal at the Gamow peak E0.
  3. E0 = (E_G (kT)^2 / 4)^(1/3): it rises with charge (E_G ~ (Z1 Z2)^2) and with
     temperature.
  4. Higher-charge reactions (C+C) need far higher temperatures to reach usable peaks.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gamow import (reduced_mass, gamow_energy, gamow_peak_energy,  # noqa: E402
                   gamow_peak_width, integrand, peak_energy_kev, M_P, K_B, KEV)

MU_PP = reduced_mass(M_P, M_P)
T_SUN = 1.5e7


def test_pp_peak_6kev():
    E0 = peak_energy_kev(1, 1, MU_PP, T_SUN)
    assert 4.0 < E0 < 8.0, f"p-p Gamow peak {E0} keV not ~6"


def test_peak_above_thermal():
    E0 = gamow_peak_energy(1, 1, MU_PP, T_SUN)
    kT = K_B * T_SUN
    assert E0 > 3.0 * kT, "the Gamow peak sits well out on the thermal tail"


def test_integrand_maximal_at_peak():
    E0 = gamow_peak_energy(1, 1, MU_PP, T_SUN)
    i0 = integrand(E0, 1, 1, MU_PP, T_SUN)
    assert i0 > integrand(0.5 * E0, 1, 1, MU_PP, T_SUN)
    assert i0 > integrand(2.0 * E0, 1, 1, MU_PP, T_SUN)


def test_peak_rises_with_temperature():
    lo = gamow_peak_energy(1, 1, MU_PP, 1e7)
    hi = gamow_peak_energy(1, 1, MU_PP, 2e7)
    assert hi > lo
    # E0 ~ T^(2/3): doubling T raises E0 by 2^(2/3)
    assert abs(hi / lo - 2.0 ** (2.0 / 3.0)) < 1e-6


def test_gamow_energy_charge_scaling():
    # E_G ~ (Z1 Z2)^2, so He-He (Z=2) has 16x the p-p Gamow energy at equal mu
    e_pp = gamow_energy(1, 1, MU_PP)
    e_22 = gamow_energy(2, 2, MU_PP)
    assert abs(e_22 / e_pp - 16.0) < 1e-6


def test_higher_charge_needs_hotter():
    mu_cc = reduced_mass(12 * M_P, 12 * M_P)
    # at the Sun's core temperature the C+C peak is far above the p-p peak
    assert peak_energy_kev(6, 6, mu_cc, T_SUN) > 10.0 * peak_energy_kev(1, 1, MU_PP, T_SUN)


def test_width_positive_and_scales():
    w1 = gamow_peak_width(1, 1, MU_PP, 1e7)
    w2 = gamow_peak_width(1, 1, MU_PP, 2e7)
    assert w1 > 0 and w2 > w1


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

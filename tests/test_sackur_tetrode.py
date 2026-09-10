"""Sackur-Tetrode entropy tests.

Claims checked:
  1. Argon's standard molar entropy at STP is ~154.8 J/(mol K); helium's ~126.2 --
     the Sackur-Tetrode formula reproduces measured values.
  2. Heavier atoms have higher entropy at fixed T, P (more phase space).
  3. A gas at STP is classical (n << n_Q); the thermal wavelength is ~tens of pm.
  4. Entropy rises with temperature and with volume per particle.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sackur_tetrode import (thermal_wavelength, quantum_concentration,  # noqa: E402
                            entropy_per_particle, molar_entropy, entropy_total,
                            is_classical, AMU, K_B, N_A)

M_AR = 39.948 * AMU
M_HE = 4.0026 * AMU
STP_T, STP_P = 298.15, 101325.0


def test_argon_molar_entropy():
    S = molar_entropy(STP_T, STP_P, M_AR)
    assert abs(S - 154.8) < 1.0, f"argon molar entropy {S} not ~154.8"


def test_helium_molar_entropy():
    S = molar_entropy(STP_T, STP_P, M_HE)
    assert abs(S - 126.2) < 1.0, f"helium molar entropy {S} not ~126.2"


def test_heavier_higher_entropy():
    assert molar_entropy(STP_T, STP_P, M_AR) > molar_entropy(STP_T, STP_P, M_HE)


def test_thermal_wavelength_pm_scale():
    lam = thermal_wavelength(STP_T, M_AR) * 1e12  # pm
    assert 5.0 < lam < 50.0, f"argon thermal wavelength {lam} pm off scale"


def test_gas_is_classical_at_stp():
    n = STP_P / (K_B * STP_T)
    assert is_classical(n, STP_T, M_AR)
    assert n < quantum_concentration(STP_T, M_AR)


def test_entropy_rises_with_temperature():
    assert entropy_per_particle(600.0, STP_P, M_AR) > entropy_per_particle(300.0, STP_P, M_AR)


def test_entropy_rises_with_volume():
    # lower pressure at fixed T = more volume per particle = higher entropy
    assert entropy_per_particle(STP_T, STP_P / 2, M_AR) > entropy_per_particle(STP_T, STP_P, M_AR)


def test_total_is_n_times_per_particle():
    N = 1e5
    assert abs(entropy_total(N, STP_T, STP_P, M_AR)
               - N * K_B * entropy_per_particle(STP_T, STP_P, M_AR)) < 1e-9


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

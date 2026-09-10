"""Quantum-statistics tests.

Claims checked:
  1. Fermi-Dirac occupation is 0.5 at E=mu, a sharp 1->0 step at T=0 (Pauli exclusion,
     never above 1).
  2. Bose-Einstein occupation diverges as E->mu and has no upper bound.
  3. Far above mu (E-mu >> kT) all three distributions converge to the classical
     Maxwell-Boltzmann exponential.
  4. A photon mode (mu=0) at E=kT has occupation 1/(e-1) ~ 0.582 (the Planck factor).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_stats import (fermi_dirac, bose_einstein, maxwell_boltzmann,  # noqa: E402
                           photon_occupation, fermi_step_width, is_classical,
                           K_B, EV)

MU = 5 * EV
T = 300.0


def test_fd_half_at_mu():
    assert abs(fermi_dirac(MU, MU, T) - 0.5) < 1e-12


def test_fd_step_at_zero_T():
    assert fermi_dirac(4 * EV, MU, 0.0) == 1.0
    assert fermi_dirac(6 * EV, MU, 0.0) == 0.0


def test_fd_never_exceeds_one():
    for E in (0.0, MU - EV, MU, MU + EV, 10 * EV):
        assert 0.0 <= fermi_dirac(E, MU, T) <= 1.0


def test_be_diverges_near_mu():
    assert bose_einstein(MU + 1e-25, MU, T) > 1000.0
    assert bose_einstein(MU, MU, T) == float("inf")


def test_classical_limit():
    E = MU + 0.5 * EV   # well above mu
    fd = fermi_dirac(E, MU, T)
    be = bose_einstein(E, MU, T)
    mb = maxwell_boltzmann(E, MU, T)
    assert abs(fd / mb - 1.0) < 0.01
    assert abs(be / mb - 1.0) < 0.01


def test_photon_planck_factor():
    kT = K_B * T
    assert abs(photon_occupation(kT, T) - 1.0 / (math.e - 1.0)) < 1e-9


def test_fermi_step_width():
    w = fermi_step_width(300.0) / EV * 1000.0  # meV
    assert 80.0 < w < 130.0


def test_is_classical():
    assert is_classical(MU + EV, MU, T)          # sparse -> classical
    assert not is_classical(MU - EV, MU, T)      # below mu, densely filled


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

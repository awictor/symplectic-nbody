"""Toomre-Q disk-stability tests.

Claims checked:
  1. The solar neighbourhood is marginally stable: stellar and gas Q ~ 1-2.
  2. Q > 1 counts as stable, Q < 1 unstable (fragments into clumps/arms).
  3. Q scales with velocity dispersion and epicyclic frequency and inversely with
     surface density; a colder or denser disk is less stable.
  4. The Toomre wavelength is a galactic-scale length (~kpc) and grows with Sigma.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from toomre import (toomre_q_gas, toomre_q_stars, is_stable,  # noqa: E402
                    toomre_wavelength, critical_dispersion_gas,
                    epicyclic_frequency_flat, G, PC, M_SUN)

KAPPA = epicyclic_frequency_flat(220e3, 8000 * PC)   # solar neighbourhood
SIGMA_STARS = 50 * M_SUN / PC ** 2
SIGMA_GAS = 13 * M_SUN / PC ** 2


def test_solar_neighbourhood_marginally_stable():
    Qs = toomre_q_stars(30e3, KAPPA, SIGMA_STARS)
    Qg = toomre_q_gas(8e3, KAPPA, SIGMA_GAS)
    assert 1.0 < Qs < 2.5, f"stellar Q {Qs} not marginally stable"
    assert 1.0 < Qg < 2.5, f"gas Q {Qg} not marginally stable"


def test_stability_verdict():
    assert is_stable(1.5)
    assert is_stable(1.0)
    assert not is_stable(0.7)


def test_q_scales_with_dispersion():
    base = toomre_q_gas(8e3, KAPPA, SIGMA_GAS)
    hot = toomre_q_gas(16e3, KAPPA, SIGMA_GAS)
    assert abs(hot / base - 2.0) < 1e-9, "Q should scale linearly with c_s"


def test_denser_disk_less_stable():
    base = toomre_q_gas(8e3, KAPPA, SIGMA_GAS)
    dense = toomre_q_gas(8e3, KAPPA, 2 * SIGMA_GAS)
    assert dense < base, "a denser disk should have a lower Q"
    assert abs(dense / base - 0.5) < 1e-9


def test_critical_dispersion_gives_q_one():
    c_crit = critical_dispersion_gas(KAPPA, SIGMA_GAS)
    Q = toomre_q_gas(c_crit, KAPPA, SIGMA_GAS)
    assert abs(Q - 1.0) < 1e-9


def test_toomre_wavelength_kpc_scale():
    lam = toomre_wavelength(SIGMA_GAS, KAPPA) / PC / 1000.0  # kpc
    assert 0.3 < lam < 5.0, f"Toomre wavelength {lam} kpc off galactic scale"


def test_wavelength_grows_with_sigma():
    l1 = toomre_wavelength(SIGMA_GAS, KAPPA)
    l2 = toomre_wavelength(2 * SIGMA_GAS, KAPPA)
    assert abs(l2 / l1 - 2.0) < 1e-9


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

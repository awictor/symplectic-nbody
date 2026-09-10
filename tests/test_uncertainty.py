"""Heisenberg-uncertainty tests.

Claims checked:
  1. dx dp = hbar/2 at the bound; position and momentum spreads invert.
  2. Confining an electron to an atom (~0.1 nm) gives an eV-scale zero-point energy;
     confining a nucleon to a nucleus (~fm) gives MeV-scale.
  3. Minimizing the confinement energy plus Coulomb potential reproduces hydrogen's
     ~13.6 eV binding from the uncertainty principle alone.
  4. The energy-time bound gives a natural line width that shrinks with lifetime.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uncertainty import (min_momentum_spread, min_position_spread,  # noqa: E402
                         confinement_energy, energy_time_bound,
                         natural_linewidth_hz, hydrogen_ground_state_estimate,
                         HBAR, M_E, M_P, EV, MEV, FM)


def test_dx_dp_bound():
    dx = 1e-10
    assert abs(dx * min_momentum_spread(dx) - HBAR / 2.0) < 1e-40


def test_position_momentum_invert():
    dp = 1e-24
    assert abs(min_position_spread(min_momentum_spread(1e-10)) - 1e-10) < 1e-20


def test_electron_atomic_scale():
    E = confinement_energy(1e-10) / EV
    assert 0.3 < E < 5.0, f"electron confinement energy {E} eV not atomic scale"


def test_nucleon_nuclear_scale():
    E = confinement_energy(3 * FM, M_P) / MEV
    assert 0.1 < E < 10.0, f"nucleon confinement energy {E} MeV not nuclear scale"


def test_hydrogen_from_uncertainty():
    E = hydrogen_ground_state_estimate() / EV
    assert abs(E - 13.6) < 0.1, f"H ground estimate {E} eV not ~13.6"


def test_tighter_confinement_more_energy():
    assert confinement_energy(5e-11) > confinement_energy(1e-10)


def test_energy_time_bound():
    dt = 1e-9
    assert abs(energy_time_bound(dt) - HBAR / (2.0 * dt)) < 1e-40


def test_linewidth_shrinks_with_lifetime():
    assert natural_linewidth_hz(1e-6) < natural_linewidth_hz(1e-9)


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

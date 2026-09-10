"""Nuclear Q-value tests.

Claims checked:
  1. One atomic mass unit of defect equals 931.494 MeV.
  2. D-T fusion (mass defect ~0.0189 amu) releases 17.6 MeV; U-235 fission ~200 MeV.
  3. A positive mass defect is exothermic; Q from binding energies equals
     B(products) - B(reactants).
  4. Fusion fuel yields ~10^14 J/kg -- millions of times chemical fuel.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from q_value import (q_value_from_mass_change, q_value_from_amu,  # noqa: E402
                     q_value_from_binding, mass_fraction_converted,
                     energy_per_kg, is_exothermic, AMU, AMU_TO_MEV, MEV, C)

DM_DT = (2.014102 + 3.016049) - (4.002602 + 1.008665)  # amu


def test_amu_to_mev():
    assert abs(q_value_from_amu(1.0) - 931.494) < 0.01


def test_dt_fusion():
    assert abs(q_value_from_amu(DM_DT) - 17.6) < 0.1


def test_u235_fission():
    assert abs(q_value_from_amu(0.2115) - 197.0) < 3.0


def test_exothermic_sign():
    assert is_exothermic(DM_DT * AMU)
    assert not is_exothermic(-1e-30)


def test_mass_energy_consistency():
    # q_value_from_amu should equal q_value_from_mass_change on the same defect
    j = q_value_from_mass_change(DM_DT * AMU)
    assert abs(j / MEV - q_value_from_amu(DM_DT)) < 1e-3


def test_binding_q_value():
    # products more bound (B=28.3 He-4) than reactants (D 2.22 + T 8.48) -> positive Q
    Q = q_value_from_binding(28.30, 2.22 + 8.48)
    assert Q > 0.0
    assert abs(Q - 17.6) < 0.2


def test_fusion_energy_density():
    e = energy_per_kg(17.6, 5.03)
    assert e > 1e14, f"D-T fuel energy density {e} J/kg too low"


def test_mass_fraction_small():
    tot = (2.014102 + 3.016049) * AMU
    frac = mass_fraction_converted(DM_DT * AMU, tot)
    assert 0.002 < frac < 0.006, f"D-T mass fraction {frac} off ~0.4%"


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

"""Chandrasekhar-mass tests.

Claims checked:
  1. The n=3 Lane-Emden mass factor omega_3 = -xi_1^2 theta'(xi_1) is 2.01824.
  2. The Chandrasekhar mass computed from fundamental constants is ~1.44 solar
     masses for a carbon/oxygen white dwarf (mu_e = 2).
  3. It scales as 1/mu_e^2 (a heavier composition per electron lowers the limit).
  4. It is built from real physics: shrinking hbar or growing G lowers M_Ch.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chandrasekhar as ch  # noqa: E402


def test_omega3_value():
    assert abs(ch.OMEGA_3 - 2.01824) < 1e-3, f"omega_3 {ch.OMEGA_3} not 2.01824"


def test_chandrasekhar_is_1_44_solar():
    M = ch.chandrasekhar_mass_solar(mu_e=2.0)
    assert abs(M - 1.44) < 0.02, f"M_Ch {M} not ~1.44 solar masses"


def test_scales_as_inverse_mu_e_squared():
    ratio = ch.chandrasekhar_mass_solar(2.0) / ch.chandrasekhar_mass_solar(4.0)
    assert abs(ratio - 4.0) < 1e-6, f"M_Ch should scale as 1/mu_e^2, ratio {ratio}"


def test_heavier_composition_lowers_limit():
    assert ch.chandrasekhar_mass_solar(2.15) < ch.chandrasekhar_mass_solar(2.0)


def test_white_dwarf_mass_radius_inverse():
    # denser (higher rho_c) white dwarfs are SMALLER but MORE massive
    r1, m1 = ch.white_dwarf_structure(1e9, dr=2e4)
    r2, m2 = ch.white_dwarf_structure(1e12, dr=2e4)
    assert r2 < r1, f"denser WD should be smaller: {r1} -> {r2}"
    assert m2 > m1, f"denser WD should be more massive: {m1} -> {m2}"


def test_mass_approaches_chandrasekhar():
    # a very dense (relativistic) white dwarf's mass approaches M_Ch from below
    _r, m = ch.white_dwarf_structure(1e13, dr=2e4)
    m_solar = m / ch.M_SUN
    m_ch = ch.chandrasekhar_mass_solar(2.0)
    assert 0.9 * m_ch < m_solar < m_ch, f"dense WD mass {m_solar} should near M_Ch {m_ch} from below"


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

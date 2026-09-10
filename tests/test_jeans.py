"""Jeans-instability tests.

Claims checked:
  1. The dispersion relation omega^2 = c_s^2 k^2 - 4 pi G rho0 splits into a
     stable branch (short wavelength, k > k_J) and an unstable branch (long
     wavelength, k < k_J).
  2. The marginal mode has exactly omega^2 = 0 at k = k_J.
  3. The growth rate of unstable modes rises as k -> 0, approaching
     sqrt(4 pi G rho0).
  4. Denser gas has a smaller Jeans length and a colder (lower c_s) gas too --
     both make collapse easier.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jeans import (jeans_wavenumber, jeans_length, jeans_mass,  # noqa: E402
                   omega_squared, is_unstable, growth_rate, free_fall_time)


def test_stable_and_unstable_branches():
    cs, rho0, G = 1.0, 1.0, 1.0
    kJ = jeans_wavenumber(cs, rho0, G)
    assert not is_unstable(2.0 * kJ, cs, rho0, G), "short wavelength should be stable"
    assert is_unstable(0.5 * kJ, cs, rho0, G), "long wavelength should collapse"


def test_marginal_mode_is_zero():
    cs, rho0, G = 1.3, 0.7, 1.0
    kJ = jeans_wavenumber(cs, rho0, G)
    assert abs(omega_squared(kJ, cs, rho0, G)) < 1e-9, "omega^2 should vanish at k_J"


def test_growth_rate_limit():
    cs, rho0, G = 1.0, 1.0, 1.0
    g = growth_rate(1e-6, cs, rho0, G)
    assert abs(g - math.sqrt(4.0 * math.pi * G * rho0)) < 1e-3, "k->0 growth wrong"


def test_growth_rate_increases_toward_long_wavelength():
    cs, rho0, G = 1.0, 1.0, 1.0
    kJ = jeans_wavenumber(cs, rho0, G)
    g_long = growth_rate(0.1 * kJ, cs, rho0, G)
    g_short = growth_rate(0.9 * kJ, cs, rho0, G)
    assert g_long > g_short > 0.0, "longer wavelengths should grow faster"


def test_denser_and_colder_lowers_jeans_length():
    base = jeans_length(1.0, 1.0)
    assert jeans_length(1.0, 4.0) < base, "denser gas -> smaller Jeans length"
    assert jeans_length(0.5, 1.0) < base, "colder gas -> smaller Jeans length"


def test_free_fall_time_scaling():
    # t_ff ~ 1/sqrt(rho); quadrupling density halves it
    assert abs(free_fall_time(4.0) - 0.5 * free_fall_time(1.0)) < 1e-9


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

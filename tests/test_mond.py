"""MOND (Modified Newtonian Dynamics) tests.

Claims checked:
  1. The interpolating functions have the right limits: mu(x) -> 1 for x >> 1
     (Newtonian) and mu(x) -> x for x << 1 (deep MOND).
  2. Solving g mu(g/a0) = g_N recovers g ~ g_N when g_N >> a0 and
     g ~ sqrt(g_N a0) when g_N << a0.
  3. A MOND rotation curve FLATTENS at large radius toward v = (G M a0)^{1/4},
     with no dark matter.
  4. The baryonic Tully-Fisher relation holds: v_flat^4 = G M a0 (v ~ M^{1/4}),
     and the mass inversion is exact.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mond import (mu_standard, mu_simple, mond_acceleration, circular_speed,  # noqa: E402
                  deep_mond_vflat, tully_fisher_mass, A0, G, M_SUN, KPC)


def test_mu_limits():
    for mu in (mu_standard, mu_simple):
        assert abs(mu(1e6) - 1.0) < 1e-3, "mu -> 1 for large x"
        assert abs(mu(1e-6) - 1e-6) / 1e-6 < 1e-2, "mu -> x for small x"


def test_high_acceleration_is_newtonian():
    g_N = 1e6 * A0
    g = mond_acceleration(g_N)
    assert abs(g - g_N) / g_N < 1e-3, "g should equal g_N in the Newtonian regime"


def test_deep_mond_sqrt_law():
    g_N = 1e-6 * A0
    g = mond_acceleration(g_N)
    expected = math.sqrt(g_N * A0)
    assert abs(g - expected) / expected < 1e-2, "deep MOND: g -> sqrt(g_N a0)"


def test_rotation_curve_flattens():
    M = 6e10 * M_SUN
    v_far = circular_speed(M, 100 * KPC)
    v_vfar = circular_speed(M, 300 * KPC)
    vflat = deep_mond_vflat(M)
    # both far points sit close to the asymptotic flat speed
    assert abs(v_far - vflat) / vflat < 0.05
    assert abs(v_vfar - vflat) / vflat < 0.02
    # and it is genuinely flat: outer speed barely changes
    assert abs(v_vfar - v_far) / v_far < 0.05


def test_baryonic_tully_fisher():
    M = 6e10 * M_SUN
    vflat = deep_mond_vflat(M)
    # v^4 = G M a0
    assert abs(vflat ** 4 - G * M * A0) / (G * M * A0) < 1e-9
    # mass inversion is exact
    assert abs(tully_fisher_mass(vflat) - M) / M < 1e-9
    # slope: doubling mass raises v by 2^(1/4)
    assert abs(deep_mond_vflat(2 * M) / vflat - 2 ** 0.25) < 1e-9


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

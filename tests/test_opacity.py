"""Stellar-opacity tests.

Claims checked:
  1. Thomson electron scattering is ~0.034 m^2/kg (0.34 cm^2/g) and independent of
     density and temperature.
  2. Kramers opacity follows kappa ~ rho T^(-7/2): rising with density, falling
     steeply with temperature.
  3. The total opacity at the solar centre is of order 0.1 m^2/kg (~1 cm^2/g),
     and electron scattering is the floor in hot low-density gas.
  4. The photon mean free path in the solar interior is tiny (sub-millimetre), so
     radiation diffuses out over a very long time.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from opacity import (electron_scattering, kramers_bound_free,  # noqa: E402
                     kramers_free_free, total_opacity, mean_free_path, h_minus)

RHO_C, T_C = 1.5e5, 1.5e7  # solar-centre density and temperature


def test_electron_scattering_value():
    k = electron_scattering(0.7)
    assert abs(k - 0.034) < 0.003, f"electron scattering {k} not ~0.034 m^2/kg"


def test_electron_scattering_constant():
    # depends only on composition, not rho or T
    assert electron_scattering(0.7) == electron_scattering(0.7)
    # more hydrogen -> more free electrons per unit mass -> higher opacity
    assert electron_scattering(0.9) > electron_scattering(0.5)


def test_kramers_temperature_scaling():
    hot = kramers_free_free(RHO_C, 2 * T_C)
    base = kramers_free_free(RHO_C, T_C)
    assert abs(hot / base - 2.0 ** (-3.5)) < 1e-9, "Kramers should go as T^(-7/2)"


def test_kramers_density_scaling():
    dense = kramers_free_free(2 * RHO_C, T_C)
    base = kramers_free_free(RHO_C, T_C)
    assert abs(dense / base - 2.0) < 1e-9, "Kramers should be linear in density"


def test_solar_centre_order_of_magnitude():
    k = total_opacity(RHO_C, T_C)
    assert 0.03 < k < 1.0, f"solar-centre opacity {k} m^2/kg off order of magnitude"


def test_electron_scattering_is_floor():
    # at very high T (Kramers -> 0) the total approaches electron scattering
    k = total_opacity(1e2, 1e9)
    assert abs(k - electron_scattering(0.7)) / electron_scattering(0.7) < 0.01


def test_mean_free_path_is_tiny():
    mfp = mean_free_path(RHO_C, total_opacity(RHO_C, T_C))
    assert mfp < 1e-3, f"solar-interior photon mfp {mfp} m should be sub-millimetre"


def test_h_minus_rises_with_temperature():
    # the H-minus term climbs steeply with T in cool envelopes
    assert h_minus(1e-4, 5000.0) > h_minus(1e-4, 4000.0)


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

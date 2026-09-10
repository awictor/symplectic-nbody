"""Friedmann-cosmology tests.

Claims checked:
  1. A matter-only universe expands as a(t) ~ t^{2/3}.
  2. A radiation-only universe expands as a(t) ~ t^{1/2}.
  3. A dark-energy-only universe expands exponentially: a grows by e per Hubble
     time (constant H).
  4. The age of a flat LCDM universe (Om=0.3, OL=0.7) is ~0.96/H0 -- which is the
     ~13.8 Gyr age for the measured Hubble constant.
  5. The density parameters plus curvature sum to 1 by construction.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from friedmann import Cosmology, local_slope  # noqa: E402


def test_matter_era_scaling():
    c = Cosmology(Omega_r=0.0, Omega_m=1.0, Omega_L=0.0)
    ts, a = c.integrate_forward(a0=1e-3, t_max=2.0, dt=1e-4)
    slope = local_slope(ts, a, len(ts) // 2)
    assert abs(slope - 2.0 / 3.0) < 0.02, f"matter slope {slope} not ~2/3"


def test_radiation_era_scaling():
    c = Cosmology(Omega_r=1.0, Omega_m=0.0, Omega_L=0.0)
    ts, a = c.integrate_forward(a0=1e-3, t_max=2.0, dt=1e-4)
    slope = local_slope(ts, a, len(ts) // 2)
    assert abs(slope - 0.5) < 0.02, f"radiation slope {slope} not ~1/2"


def test_lambda_era_is_exponential():
    c = Cosmology(Omega_r=0.0, Omega_m=0.0, Omega_L=1.0, H0=1.0)
    ts, a = c.integrate_forward(a0=1.0, t_max=3.0, dt=1e-4)
    # find sample nearest t=1; a should be ~e times a(0)
    i = min(range(len(ts)), key=lambda k: abs(ts[k] - 1.0))
    assert abs(a[i] / a[0] - math.e) < 0.01, f"Lambda growth {a[i]/a[0]} not ~e"


def test_lcdm_age():
    c = Cosmology(Omega_r=0.0, Omega_m=0.3, Omega_L=0.7)
    age = c.age()
    assert abs(age - 0.964) < 0.02, f"LCDM age {age} not ~0.96 Hubble times"


def test_density_parameters_sum_to_one():
    c = Cosmology(Omega_r=1e-4, Omega_m=0.3, Omega_L=0.7)
    total = c.Omega_r + c.Omega_m + c.Omega_k + c.Omega_L
    assert abs(total - 1.0) < 1e-12


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

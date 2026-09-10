"""Relativistic-rocket tests.

Claims checked:
  1. At 1 g the ship reaches ~0.77 c after one year of proper time, gamma ~ 1.58.
  2. Velocity approaches but never reaches c; gamma and Earth time grow without bound.
  3. Proper time to cover a distance inverts the distance formula, and the galactic
     centre (27,000 ly) is reachable in ~10 yr of ship time (accel only) while ~27,000
     yr pass on Earth.
  4. The photon-rocket mass ratio grows exponentially with rapidity.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relativistic_rocket import (velocity, lorentz_gamma, distance,  # noqa: E402
                                 earth_time, proper_time_for_distance,
                                 photon_rocket_mass_ratio, rapidity,
                                 C, G_EARTH, YEAR, LY)


def test_one_year_at_1g():
    v = velocity(G_EARTH, YEAR) / C
    assert abs(v - 0.775) < 0.005
    assert abs(lorentz_gamma(G_EARTH, YEAR) - 1.582) < 0.01


def test_never_reaches_c():
    # v = c tanh(...) is strictly below c until float tanh saturates to 1.0 at
    # rapidity ~19; check the physical (non-saturated) regime
    for tau in (0.5 * YEAR, YEAR, 5 * YEAR, 10 * YEAR):
        assert velocity(G_EARTH, tau) < C


def test_distance_and_proper_time_invert():
    d = 27000 * LY
    tau = proper_time_for_distance(G_EARTH, d)
    assert abs(distance(G_EARTH, tau) - d) / d < 1e-9


def test_galactic_centre_ship_time():
    tau = proper_time_for_distance(G_EARTH, 27000 * LY) / YEAR
    assert 8.0 < tau < 14.0, f"galactic-centre ship time {tau} yr off scale"


def test_earth_time_far_ahead():
    tau = proper_time_for_distance(G_EARTH, 27000 * LY)
    t_earth = earth_time(G_EARTH, tau) / YEAR
    assert t_earth > 25000.0   # dwarfs the ~10 yr of ship time


def test_gamma_grows():
    assert lorentz_gamma(G_EARTH, 10 * YEAR) > lorentz_gamma(G_EARTH, YEAR)


def test_mass_ratio_exponential():
    r1 = photon_rocket_mass_ratio(G_EARTH, YEAR)
    r2 = photon_rocket_mass_ratio(G_EARTH, 2 * YEAR)
    # rapidity doubles -> mass ratio squares
    assert abs(r2 / r1 ** 2 - 1.0) < 1e-6


def test_rapidity_additive():
    assert abs(rapidity(G_EARTH, YEAR) - G_EARTH * YEAR / C) < 1e-9
    assert abs(velocity(G_EARTH, YEAR) / C - math.tanh(rapidity(G_EARTH, YEAR))) < 1e-9


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

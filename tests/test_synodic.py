"""Synodic-period tests.

Claims checked:
  1. Mars's synodic period seen from Earth is ~780 days (the ~26-month launch-window
     cadence); Venus ~584 d, Mercury ~116 d.
  2. The synodic (new-Moon-to-new-Moon) month is ~29.5 days, longer than the 27.3-day
     sidereal month.
  3. Nearly equal periods give an enormous synodic period; equal periods give infinity.
  4. Faster inner planets lap Earth more often (more conjunctions per year).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synodic import (synodic_period, synodic_from_earth, relative_angular_rate,  # noqa: E402
                     conjunctions_per_year, synodic_month, P_MERCURY, P_VENUS,
                     P_EARTH, P_MARS, P_JUPITER, P_MOON_SIDEREAL)


def test_mars_synodic():
    assert abs(synodic_from_earth(P_MARS) - 780.0) < 2.0


def test_venus_and_mercury():
    assert abs(synodic_from_earth(P_VENUS) - 584.0) < 2.0
    assert abs(synodic_from_earth(P_MERCURY) - 116.0) < 1.0


def test_synodic_month():
    S = synodic_month()
    assert abs(S - 29.53) < 0.05
    assert S > P_MOON_SIDEREAL   # synodic longer than sidereal


def test_equal_periods_infinite():
    assert synodic_period(365.0, 365.0) == float("inf")


def test_near_equal_large():
    assert synodic_period(365.0, 366.0) > 1e4


def test_inner_planets_more_conjunctions():
    # Mercury laps Earth more often than Jupiter does
    assert conjunctions_per_year(P_MERCURY, P_EARTH) > conjunctions_per_year(P_JUPITER, P_EARTH)


def test_outer_planet_approaches_one_year():
    # a very distant planet's synodic period -> Earth's year (Earth does the lapping)
    assert abs(synodic_from_earth(1e6) - P_EARTH) < 1.0


def test_relative_rate_matches_synodic():
    S = synodic_period(P_MARS, P_EARTH)
    rate = relative_angular_rate(P_MARS, P_EARTH)
    assert abs(2.0 * math.pi / rate - S) < 1e-6


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

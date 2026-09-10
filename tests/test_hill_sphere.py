"""Hill-sphere tests.

Claims checked:
  1. Earth's Hill radius is ~1.5 million km (about 4x the Moon's distance), so the
     Moon is comfortably bound.
  2. r_H ~ a (m/3M)^(1/3): bigger orbit or heavier planet -> bigger Hill sphere.
  3. The Moon's own Hill sphere is ~60,000 km (why it holds no sub-moons).
  4. The mutual Hill radius spaces planetary orbits; the Earth-Venus separation is
     many mutual Hill radii (dynamically stable).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hill_sphere import (hill_radius, stable_moon_limit, is_moon_stable,  # noqa: E402
                         mutual_hill_radius, separation_in_mutual_hill,
                         AU, M_SUN, M_EARTH)


def test_earth_hill_radius():
    rH = hill_radius(AU, M_EARTH, M_SUN) / 1e9  # million km
    assert abs(rH - 1.5) < 0.1, f"Earth Hill radius {rH} Mkm not ~1.5"


def test_moon_is_bound():
    assert is_moon_stable(3.84e8, AU, M_EARTH, M_SUN)


def test_distant_moon_unstable():
    # a moon out near the full Hill radius is beyond the ~1/2 r_H stable limit
    rH = hill_radius(AU, M_EARTH, M_SUN)
    assert not is_moon_stable(0.9 * rH, AU, M_EARTH, M_SUN)


def test_scales_with_orbit():
    r1 = hill_radius(AU, M_EARTH, M_SUN)
    r2 = hill_radius(2 * AU, M_EARTH, M_SUN)
    assert abs(r2 / r1 - 2.0) < 1e-9


def test_scales_with_mass_cube_root():
    r1 = hill_radius(AU, M_EARTH, M_SUN)
    r8 = hill_radius(AU, 8 * M_EARTH, M_SUN)
    assert abs(r8 / r1 - 2.0) < 1e-9   # 8^(1/3) = 2


def test_moon_hill_sphere():
    m_moon = 7.342e22
    rH = hill_radius(3.84e8, m_moon, M_EARTH) / 1e3  # km
    assert 50000.0 < rH < 70000.0, f"Moon Hill radius {rH} km not ~60000"


def test_eccentric_smaller():
    circ = hill_radius(AU, M_EARTH, M_SUN, e=0.0)
    ecc = hill_radius(AU, M_EARTH, M_SUN, e=0.3)
    assert abs(ecc / circ - 0.7) < 1e-9


def test_mutual_hill_spacing():
    sep = separation_in_mutual_hill(0.72 * AU, AU, 0.815 * M_EARTH, M_EARTH, M_SUN)
    assert sep > 10.0, f"Earth-Venus spacing {sep} mutual Hill radii should be stable"


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

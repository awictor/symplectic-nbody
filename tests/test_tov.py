"""Tolman-Oppenheimer-Volkoff (neutron-star) tests.

Claims checked:
  1. A polytropic neutron star has neutron-star-scale structure: R ~ 10 km and
     M ~ 1 solar mass for a plausible central density.
  2. The TOV mass-radius sequence has a MAXIMUM mass (it turns over) -- the GR
     result that no static neutron star can exceed a critical mass.
  3. The Newtonian version has NO maximum: its mass grows without bound as the
     central density rises. GR is what imposes the limit.
  4. Higher central density past the peak gives a SMALLER, less massive star
     (the unstable branch).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tov import integrate_star, mass_radius_sequence, maximum_mass  # noqa: E402

K, GAMMA = 100.0, 2.0
RHO_CS = [10 ** (-3 + 0.15 * i) for i in range(20)]


def test_neutron_star_scale():
    R, M = integrate_star(1e-3, K, GAMMA, dr=1e-3)
    assert 8.0 < R < 15.0, f"NS radius {R} km not neutron-star scale"
    assert 0.5 < M < 1.5, f"NS mass {M} not ~1 solar mass"


def test_tov_has_maximum_mass():
    seq = mass_radius_sequence(RHO_CS, K, GAMMA)
    m_max = maximum_mass(seq)
    # the sequence turns over: the last (densest) star is less massive than the peak
    assert seq[-1][1] < m_max, "TOV sequence should turn over (have a max mass)"
    assert 1.0 < m_max < 2.5, f"TOV max mass {m_max} not in the expected range"


def test_newtonian_has_no_maximum():
    seq_n = mass_radius_sequence(RHO_CS, K, GAMMA, newtonian=True)
    # Newtonian mass keeps rising with central density -- densest is the heaviest
    assert seq_n[-1][1] == maximum_mass(seq_n), "Newtonian mass should not turn over"
    assert seq_n[-1][1] > 10.0, "Newtonian star grows unbounded, well past any GR limit"


def test_gr_limits_mass_below_newtonian():
    seq_t = mass_radius_sequence(RHO_CS, K, GAMMA)
    seq_n = mass_radius_sequence(RHO_CS, K, GAMMA, newtonian=True)
    assert maximum_mass(seq_t) < maximum_mass(seq_n), "GR should cap the mass below Newtonian"


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

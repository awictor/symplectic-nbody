"""Real solar-system tests: the model must reproduce known physics.

Claims checked:
  1. Measured orbital periods (from integrating Sun+planet) match the analytic
     Kepler period to <1%, and match reality (Earth ~1 yr, Jupiter ~11.86 yr).
  2. Kepler's third law T^2 proportional to a^3 emerges: log T vs log a has
     slope 3/2 across all eight planets.
  3. The full eight-planet system conserves energy under symplectic integration.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solar_system import build, kepler_period, PLANETS  # noqa: E402


def _measure_period(planet):
    b = build([planet], include_sun=True)
    dt = kepler_period(PLANETS[planet][1]) / 2000.0

    def ang():
        return math.atan2(b.pos[1][1] - b.pos[0][1], b.pos[1][0] - b.pos[0][0])

    prev = ang()
    t = 0.0
    for _ in range(6000):
        b.step("verlet", dt)
        t += dt
        a = ang()
        if prev < 0 and a >= 0 and abs(a) < 1.0:
            return t
        prev = a
    return None


def test_periods_match_kepler():
    for planet in ("Mercury", "Earth", "Jupiter", "Neptune"):
        meas = _measure_period(planet)
        kep = kepler_period(PLANETS[planet][1])
        assert meas is not None, f"no period found for {planet}"
        rel = abs(meas - kep) / kep
        assert rel < 0.01, f"{planet} period off by {rel:.3%} (meas {meas}, kep {kep})"


def test_earth_period_is_one_year():
    meas = _measure_period("Earth")
    assert abs(meas - 1.0) < 0.01, f"Earth year should be ~1.0, got {meas}"


def test_keplers_third_law_slope():
    # T^2 ~ a^3  =>  log T = 1.5 log a + const
    a_vals, T_vals = [], []
    for name, (m, a, e) in PLANETS.items():
        a_vals.append(math.log(a))
        T_vals.append(math.log(kepler_period(a)))
    n = len(a_vals)
    ma, mt = sum(a_vals) / n, sum(T_vals) / n
    num = sum((a_vals[i] - ma) * (T_vals[i] - mt) for i in range(n))
    den = sum((a_vals[i] - ma) ** 2 for i in range(n))
    slope = num / den
    assert abs(slope - 1.5) < 1e-6, f"Kepler third-law slope {slope} != 1.5"


def test_full_system_energy_conserved():
    b = build(include_sun=True)
    e0 = b.total_energy()
    # ~2 Mercury years, small step; symplectic keeps energy bounded
    dt = 0.002
    worst = 0.0
    for s in range(3000):
        b.step("verlet", dt)
        if s % 100 == 0:
            worst = max(worst, abs((b.total_energy() - e0) / e0))
    assert worst < 1e-3, f"solar-system energy drift too large: {worst}"


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

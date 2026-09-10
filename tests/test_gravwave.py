"""Gravitational-wave inspiral tests (leading-order / Peters 1964).

Claims checked:
  1. Radiation reaction REMOVES energy: a binary's orbit shrinks, never grows.
  2. The energy-loss rate matches Peters' circular formula
     dE/dt = -(32/5) G^4 mu^2 M^3 / (c^5 a^5) to a few percent in a regime where
     the leading term dominates.
  3. The orbital frequency chirps UPWARD as the orbit shrinks.
  4. Energy loss scales as 1/c^5 (halving c increases the loss rate 32x).
  5. Turning off radiation (c -> huge) leaves a closed, non-shrinking orbit.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gravwave import inspiral, _rk4_rel  # noqa: E402

G, M, mu = 1.0, 1.0, 0.25
m1 = m2 = 0.5


def _dedt(a0, c, orbits=8):
    period = 2.0 * math.pi * math.sqrt(a0 ** 3 / M)
    dt = period / 3000
    r, v = [a0, 0.0, 0.0], [0.0, math.sqrt(M / a0), 0.0]
    E0 = 0.5 * mu * v[1] ** 2 - mu * M / a0
    for _ in range(3000 * orbits):
        r, v = _rk4_rel(r, v, m1, m2, c, dt)
    d = math.sqrt(sum(x * x for x in r))
    E1 = 0.5 * mu * sum(x * x for x in v) - mu * M / d
    return (E1 - E0) / (period * orbits), d


def test_orbit_shrinks():
    _, d = _dedt(1.0, 20.0)
    assert d < 1.0, f"radiating orbit should shrink, got separation {d}"


def test_energy_loss_matches_peters():
    c = 30.0  # leading term dominant, v/c small
    measured, _ = _dedt(1.0, c)
    peters = -(32.0 / 5.0) * mu * mu * M ** 3 / (c ** 5 * 1.0 ** 5)
    rel = abs(measured - peters) / abs(peters)
    assert rel < 0.05, f"dE/dt off from Peters by {rel:.3%} (meas {measured}, Peters {peters})"


def test_frequency_chirps_up():
    c = 30.0
    period = 2.0 * math.pi
    ts, seps, freqs = inspiral(1.0, m1, m2, c, dt=period / 2000,
                               n_steps=2000 * 60, sample_every=200)
    assert freqs[-1] > freqs[0], f"frequency should rise: {freqs[0]} -> {freqs[-1]}"
    assert seps[-1] < seps[0], "separation should shrink"


def test_loss_scales_as_c_minus_5():
    m1_, _ = _dedt(1.0, 30.0)
    m2_, _ = _dedt(1.0, 15.0)
    # halving c -> 2^5 = 32x the loss rate
    ratio = m2_ / m1_
    assert abs(ratio - 32.0) / 32.0 < 0.1, f"loss should scale 1/c^5: ratio {ratio}"


def test_no_radiation_no_shrink():
    c = 1e6  # radiation reaction negligible
    _, d = _dedt(1.0, c, orbits=20)
    assert abs(d - 1.0) < 1e-3, f"non-radiating orbit should not shrink, got {d}"


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

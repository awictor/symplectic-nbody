"""Validate PELT: recovers known change points, penalty controls count, matches brute-force DP, means."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pelt_changepoint as pc


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def piecewise(levels_lengths, noise, seed):
    rng = _R(seed)
    data = []
    truth = []
    pos = 0
    for level, length in levels_lengths:
        for _ in range(length):
            data.append(level + noise * rng.normal())
        pos += length
        truth.append(pos)
    truth = truth[:-1]  # last boundary is the end, not a change point
    return data, truth


def main():
    print("PELT change-point tests")

    # --- recover known change points of a piecewise-constant signal ---
    data, truth = piecewise([(0.0, 60), (5.0, 60), (1.0, 60)], noise=0.5, seed=1)
    res = pc.pelt(data)
    cps = res["change_points"]
    # each true change point should have a detected one within a few samples
    ok = all(any(abs(c - t) <= 3 for c in cps) for t in truth)
    check(f"recovers change points near {truth} (got {cps})", ok)
    check("correct number of segments", res["n_segments"] == 3)

    # --- larger penalty -> fewer change points ---
    r_small = pc.pelt(data, penalty=2.0)
    r_big = pc.pelt(data, penalty=200.0)
    check(f"larger penalty -> fewer change points ({len(r_small['change_points'])} >= {len(r_big['change_points'])})",
          len(r_small["change_points"]) >= len(r_big["change_points"]))

    # --- huge penalty -> single segment ---
    r_huge = pc.pelt(data, penalty=1e9)
    check("huge penalty -> one segment", r_huge["n_segments"] == 1)

    # --- tiny penalty -> many change points ---
    r_tiny = pc.pelt(data, penalty=0.01)
    check("tiny penalty -> many change points", len(r_tiny["change_points"]) > 3)

    # --- PELT matches the unpruned brute-force DP exactly ---
    for seed in (1, 2, 3):
        d, _ = piecewise([(0.0, 30), (3.0, 30), (0.0, 30)], noise=0.7, seed=seed)
        cps_pelt = pc.pelt(d)["change_points"]
        cps_brute = pc.pelt_bruteforce(d)
        check(f"PELT == brute-force DP (seed {seed})", cps_pelt == cps_brute)

    # --- matches brute force across a range of penalties ---
    d, _ = piecewise([(0.0, 40), (4.0, 40)], noise=0.6, seed=5)
    for pen in (1.0, 5.0, 20.0, 50.0):
        check(f"PELT == brute force (penalty {pen})",
              pc.pelt(d, penalty=pen)["change_points"] == pc.pelt_bruteforce(d, penalty=pen))

    # --- segment means reconstruct the levels ---
    data, truth = piecewise([(0.0, 50), (10.0, 50)], noise=0.3, seed=7)
    res = pc.pelt(data)
    means = res["segment_means"]
    check(f"segment means near true levels ({[round(m,1) for m in means]})",
          len(means) == 2 and abs(means[0]) < 0.5 and abs(means[1] - 10) < 0.5)

    # --- pure noise -> few or no change points ---
    rng = _R(9)
    noise_only = [rng.normal() for _ in range(150)]
    res = pc.pelt(noise_only)
    check(f"pure noise -> few change points ({len(res['change_points'])})",
          len(res["change_points"]) <= 3)

    # --- single change point cleanly located ---
    data, truth = piecewise([(0.0, 80), (6.0, 80)], noise=0.4, seed=11)
    cps = pc.pelt(data)["change_points"]
    check(f"single change point near {truth[0]} (got {cps})",
          len(cps) == 1 and abs(cps[0] - truth[0]) <= 2)

    # --- deterministic ---
    check("deterministic", pc.pelt(data)["change_points"] == pc.pelt(data)["change_points"])

    # --- min_size respected: no segment shorter than min_size ---
    res = pc.pelt(data, min_size=10)
    bounds = [0] + res["change_points"] + [len(data)]
    seg_lengths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    check(f"min_size respected ({min(seg_lengths)} >= 10)", min(seg_lengths) >= 10)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

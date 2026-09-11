"""Tests for perlin: lattice zeros, bounds, determinism, continuity, mean, fBm roughness."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from perlin import Perlin, _fade

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


p = Perlin(seed=1)

# --- gradient noise is zero at integer lattice points ----------------------
check("2-D noise is zero at all integer lattice points",
      all(abs(p.noise2(i, j)) < 1e-9 for i in range(-3, 8) for j in range(-3, 8)))
check("1-D noise is zero at integer lattice points",
      all(abs(p.noise1(i)) < 1e-9 for i in range(-5, 15)))

# --- bounds ----------------------------------------------------------------
vals2 = [p.noise2(i * 0.13, j * 0.17) for i in range(80) for j in range(80)]
check("2-D noise stays within [-1.5, 1.5]", all(-1.5 <= v <= 1.5 for v in vals2))
vals1 = [p.noise1(i * 0.1) for i in range(1000)]
check("1-D noise stays within [-1.5, 1.5]", all(-1.5 <= v <= 1.5 for v in vals1))

# --- determinism -----------------------------------------------------------
check("same seed, same input -> same value",
      Perlin(7).noise2(3.14, 2.71) == Perlin(7).noise2(3.14, 2.71))
check("different seeds generally differ",
      Perlin(1).noise2(3.14, 2.71) != Perlin(2).noise2(3.14, 2.71))

# --- continuity: small input change -> small output change -----------------
cont_ok = True
for _ in range(1000):
    # sample a base point deterministically
    pass
base_pts = [(i * 0.37 % 10, i * 0.53 % 10) for i in range(500)]
maxjump = 0.0
for (x, y) in base_pts:
    v0 = p.noise2(x, y)
    v1 = p.noise2(x + 1e-3, y)
    v2 = p.noise2(x, y + 1e-3)
    maxjump = max(maxjump, abs(v1 - v0), abs(v2 - v0))
check(f"noise is continuous (max jump for 1e-3 step is small: {maxjump:.5f})", maxjump < 0.05)

# --- fade curve endpoints and midpoint -------------------------------------
check("fade(0) == 0", _fade(0.0) == 0.0)
check("fade(1) == 1", abs(_fade(1.0) - 1.0) < 1e-12)
check("fade(0.5) == 0.5", abs(_fade(0.5) - 0.5) < 1e-12)
check("fade is monotone increasing", all(_fade(t / 100) <= _fade((t + 1) / 100) for t in range(100)))

# --- mean over a large region is near zero ---------------------------------
mean2 = sum(vals2) / len(vals2)
check(f"2-D noise mean is near zero ({mean2:.4f})", abs(mean2) < 0.05)
mean1 = sum(vals1) / len(vals1)
check(f"1-D noise mean is near zero ({mean1:.4f})", abs(mean1) < 0.05)

# --- noise actually varies (not constant) ----------------------------------
check("noise is non-trivial (has spread)", max(vals2) - min(vals2) > 0.5)

# --- fBm stays bounded and adds detail with more octaves -------------------
fbm_vals = [p.fbm2(i * 0.05, j * 0.05, octaves=5) for i in range(60) for j in range(60)]
check("fBm stays within [-1.2, 1.2]", all(-1.2 <= v <= 1.2 for v in fbm_vals))


def roughness(field_fn, n=2000, step=0.01):
    # second-difference magnitude = a proxy for high-frequency content; sample finely so the
    # high-frequency octaves (which oscillate fast) are actually resolved.
    total = 0.0
    a = field_fn(0.0)
    b = field_fn(step)
    for i in range(2, n):
        c = field_fn(i * step)
        total += abs(c - 2 * b + a)      # discrete second derivative
        a, b = b, c
    return total


rough_1 = roughness(lambda x: p.fbm1(x, octaves=1))
rough_6 = roughness(lambda x: p.fbm1(x, octaves=6))
check(f"more octaves add high-frequency detail ({rough_1:.3f} -> {rough_6:.3f})", rough_6 > rough_1)

# --- fBm is deterministic --------------------------------------------------
check("fBm is reproducible",
      Perlin(3).fbm2(1.1, 2.2, octaves=4) == Perlin(3).fbm2(1.1, 2.2, octaves=4))

# --- single octave fBm equals the base noise (normalized) ------------------
check("1-octave fBm equals base noise", abs(p.fbm2(2.3, 4.5, octaves=1) - p.noise2(2.3, 4.5)) < 1e-12)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all perlin tests passed")

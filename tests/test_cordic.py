"""Tests for cordic: cos/sin/atan2/hypot/exp/ln/sqrt vs math library across their ranges."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cordic import (cordic_cos, cordic_sin, cos_sin, cordic_atan2, cordic_hypot,
                    cordic_exp, cordic_ln, cordic_sqrt, _K, _N)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 456
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


# --- cos / sin across the full circle --------------------------------------
cos_ok = sin_ok = True
maxerr = 0.0
for k in range(2000):
    theta = (rng() * 4 - 2) * math.pi        # [-2pi, 2pi]
    ec = abs(cordic_cos(theta) - math.cos(theta))
    es = abs(cordic_sin(theta) - math.sin(theta))
    maxerr = max(maxerr, ec, es)
    if ec > 1e-9:
        cos_ok = False
    if es > 1e-9:
        sin_ok = False
check(f"cos accurate to 1e-9 over [-2pi,2pi] (max err {maxerr:.2e})", cos_ok)
check("sin accurate to 1e-9 over [-2pi,2pi]", sin_ok)

# --- exact values ----------------------------------------------------------
check("cos(0) = 1", abs(cordic_cos(0.0) - 1.0) < 1e-12)
check("sin(0) = 0", abs(cordic_sin(0.0)) < 1e-12)
check("cos(pi/2) = 0", abs(cordic_cos(math.pi / 2)) < 1e-9)
check("sin(pi/2) = 1", abs(cordic_sin(math.pi / 2) - 1.0) < 1e-9)
check("cos(pi) = -1", abs(cordic_cos(math.pi) + 1.0) < 1e-9)

# --- pythagorean identity --------------------------------------------------
ok = True
for _ in range(500):
    t = rng() * 10 - 5
    c, s = cos_sin(t)
    if abs(c * c + s * s - 1.0) > 1e-9:
        ok = False
        break
check("cos^2 + sin^2 = 1 everywhere", ok)

# --- atan2 across all four quadrants ---------------------------------------
ok = True
maxe = 0.0
for _ in range(1000):
    x = rng() * 4 - 2
    y = rng() * 4 - 2
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        continue
    e = abs(cordic_atan2(y, x) - math.atan2(y, x))
    maxe = max(maxe, e)
    if e > 1e-9:
        ok = False
        break
check(f"atan2 matches math.atan2 in all quadrants (max err {maxe:.2e})", ok)

# --- hypot -----------------------------------------------------------------
ok = True
for _ in range(500):
    x = rng() * 10
    y = rng() * 10
    if abs(cordic_hypot(x, y) - math.hypot(x, y)) > 1e-8:
        ok = False
        break
check("hypot matches math.hypot", ok)

# --- exp over a wide range (range reduction) -------------------------------
ok = True
maxrel = 0.0
for _ in range(500):
    x = rng() * 20 - 10        # [-10, 10]
    rel = abs(cordic_exp(x) - math.exp(x)) / math.exp(x)
    maxrel = max(maxrel, rel)
    if rel > 1e-8:
        ok = False
        break
check(f"exp matches math.exp over [-10,10] (max rel err {maxrel:.2e})", ok)
check("exp(0) = 1", abs(cordic_exp(0.0) - 1.0) < 1e-12)
check("exp(1) = e", abs(cordic_exp(1.0) - math.e) < 1e-9)

# --- ln over a wide range --------------------------------------------------
ok = True
for _ in range(500):
    a = rng() * 1000 + 1e-3
    if abs(cordic_ln(a) - math.log(a)) > 1e-8:
        ok = False
        break
check("ln matches math.log over (0, 1000]", ok)
check("ln(1) = 0", abs(cordic_ln(1.0)) < 1e-9)
check("ln(e) = 1", abs(cordic_ln(math.e) - 1.0) < 1e-9)
raised = False
try:
    cordic_ln(-1.0)
except ValueError:
    raised = True
check("ln of a non-positive raises", raised)

# --- sqrt over a wide range ------------------------------------------------
ok = True
for _ in range(500):
    a = rng() * 10000
    if abs(cordic_sqrt(a) - math.sqrt(a)) > 1e-6 * max(1.0, math.sqrt(a)):
        ok = False
        break
check("sqrt matches math.sqrt over [0, 10000]", ok)
check("sqrt(0) = 0", cordic_sqrt(0.0) == 0.0)
check("sqrt(2) accurate", abs(cordic_sqrt(2.0) - math.sqrt(2)) < 1e-9)

# --- more iterations improve accuracy (structural: gain matches analytic) --
gain_exact = 1.0
for i in range(_N):
    gain_exact *= math.sqrt(1.0 + 2.0 ** (-2 * i))
check("circular gain K matches the analytic product", abs(_K - gain_exact) < 1e-15)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all cordic tests passed")

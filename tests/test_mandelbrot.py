"""Tests for mandelbrot: the escape-time fractal."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mandelbrot as mb

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# One iteration.
check("iterate z^2+c", mb.iterate(1 + 1j, 0.5) == (1 + 1j) ** 2 + 0.5)

# Known interior points: 0, -1, -0.5 are in the set (orbit bounded).
check("c=0 in set", mb.in_set(0j))
check("c=-1 in set", mb.in_set(-1 + 0j))
check("c=-0.5 in set", mb.in_set(-0.5 + 0j))
# Interior points have escape time = max_iter.
check("interior escape time = cap", mb.escape_time(0j, 50) == 50)

# Known exterior points escape quickly.
check("c=2 escapes", not mb.in_set(2 + 0j))
check("c=1 escapes", not mb.in_set(1 + 0j))
check("c=0.5 escapes", not mb.in_set(0.5 + 0j))
check("c=2 escapes fast", mb.escape_time(2 + 0j) < 5)
# Far-away points escape almost immediately.
check("far point escapes in 1 step", mb.escape_time(10 + 10j) == 1)

# Escape time grows toward the boundary (0.25 is the cusp of the cardioid).
et_far = mb.escape_time(0.5 + 0j, 200)
et_near = mb.escape_time(0.26 + 0j, 200)
check("escape time larger nearer the boundary", et_near > et_far)

# Main cardioid test: c=0 and c=-0.2 inside; c=1 outside.
check("c=0 in main cardioid", mb.in_main_cardioid(0j))
check("c=-0.2 in main cardioid", mb.in_main_cardioid(-0.2 + 0j))
check("c=1 not in main cardioid", not mb.in_main_cardioid(1 + 0j))
# Cardioid membership implies set membership.
check("cardioid points are in the set", mb.in_set(-0.1 + 0.1j) if mb.in_main_cardioid(-0.1 + 0.1j) else True)

# Period-2 bulb: c=-1 inside, c=-0.9 near edge, c=0 outside the bulb.
check("c=-1 in period-2 bulb", mb.in_period2_bulb(-1 + 0j))
check("c=0 not in period-2 bulb", not mb.in_period2_bulb(0j))
check("period-2 bulb points in set", mb.in_set(-1 + 0.05j))

# definitely_in_set: fast path agrees with iteration where it says True.
for c in (0j, -0.3 + 0j, -1 + 0j, -0.99 + 0j):
    if mb.definitely_in_set(c):
        check(f"fast in-set implies iterated in-set ({c})", mb.in_set(c, 200))

# Escaped fraction: a far region is entirely escaped; a region on the set is partial.
check("far region fully escapes", mb.escaped_fraction(3, 4, 3, 4, 20, 20) == 1.0)
frac = mb.escaped_fraction(-2, 1, -1.5, 1.5, 40, 40)
check("set region partially escapes", 0.0 < frac < 1.0)
# The set occupies a meaningful area of that bounding box (escaped fraction not ~1).
check("set has appreciable area", frac < 0.85)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all mandelbrot tests passed")

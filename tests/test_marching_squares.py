"""Tests for marching_squares: circle contours, linear ramps, empty levels, saddle cases."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from marching_squares import (contour_segments, contour_from_function, total_length, _interp,
                              _case_segments)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- linear interpolation --------------------------------------------------
check("interp at the midpoint level", _interp((0, 0), 0, (2, 0), 4, 2) == (1.0, 0.0))
check("interp at an endpoint", _interp((0, 0), 0, (2, 0), 4, 0) == (0.0, 0.0))
check("interp equal values returns first point", _interp((0, 0), 5, (2, 0), 5, 3) == (0, 0))

# --- a radial field contours to a circle -----------------------------------
R = 3.0
segs = contour_from_function(lambda x, y: x * x + y * y, -5, 5, -5, 5, R * R, nx=120, ny=120)
check("circle contour produces segments", len(segs) > 0)
radii_err = [abs(math.hypot(p[0], p[1]) - R) for a, b in segs for p in (a, b)]
check("all contour points lie on the circle", max(radii_err) < 0.01)
check("contour length approximates 2*pi*R",
      approx(total_length(segs), 2 * math.pi * R, 0.1))

# --- contours at different levels have the right radii ---------------------
for level_r in (1.0, 2.0, 4.0):
    s = contour_from_function(lambda x, y: x * x + y * y, -6, 6, -6, 6, level_r ** 2, nx=140, ny=140)
    err = max(abs(math.hypot(p[0], p[1]) - level_r) for a, b in s for p in (a, b))
    check(f"radius-{level_r} contour is accurate", err < 0.02)

# --- levels outside the field range give no contour ------------------------
check("level below the whole field -> empty",
      contour_from_function(lambda x, y: x * x + y * y, -5, 5, -5, 5, -1.0) == [])
check("level above the whole field -> empty",
      contour_from_function(lambda x, y: x * x + y * y, -5, 5, -5, 5, 1000.0) == [])

# --- a linear ramp gives a straight contour --------------------------------
ramp = contour_from_function(lambda x, y: x, 0, 4, 0, 4, 2.0, nx=40, ny=40)
check("ramp contour is a vertical line at x=2",
      max(abs(p[0] - 2.0) for a, b in ramp for p in (a, b)) < 1e-9)
ramp_y = contour_from_function(lambda x, y: y, 0, 4, 0, 4, 1.5, nx=40, ny=40)
check("horizontal ramp contour at y=1.5",
      max(abs(p[1] - 1.5) for a, b in ramp_y for p in (a, b)) < 1e-9)

# --- single-cell cases -----------------------------------------------------
# only the top-left corner above the level -> one segment on the left/top edges
one = contour_segments([[0, 0], [0, 10]], 5)     # grid[1][0] = tl = 10
check("single high corner -> one segment", len(one) == 1)
check("segment connects the two crossed edges",
      set(one[0]) == {(0.5, 1.0), (1.0, 0.5)})
# three corners above (case 7-ish) still one segment
three = contour_segments([[10, 10], [0, 10]], 5)
check("three high corners -> one segment", len(three) == 1)
# all corners above -> no contour
check("all above -> no segment", contour_segments([[10, 10], [10, 10]], 5) == [])
check("all below -> no segment", contour_segments([[0, 0], [0, 0]], 5) == [])

# --- a saddle (two diagonal highs) gives two segments ----------------------
# case 5: bottom-left and top-right high, others low
saddle = contour_segments([[10, 0], [0, 10]], 5)   # bl=10, br=0, tr=10, tl=0
check("diagonal saddle -> two segments", len(saddle) == 2)

# --- a blob's contour forms a closed loop (endpoints pair up) --------------
blob = contour_from_function(lambda x, y: -(x * x + y * y), -4, 4, -4, 4, -4.0, nx=80, ny=80)
# every contour vertex should be shared by an even count (closed loop, no loose ends)
from collections import Counter
endpoint_counts = Counter()
for a, b in blob:
    endpoint_counts[(round(a[0], 6), round(a[1], 6))] += 1
    endpoint_counts[(round(b[0], 6), round(b[1], 6))] += 1
# a closed contour: most vertices appear an even number of times (shared by 2 segments)
odd = sum(1 for c in endpoint_counts.values() if c % 2 == 1)
check("closed blob contour has no loose ends", odd == 0)

# --- explicit-grid API with custom spacing ---------------------------------
grid = [[0, 0, 0], [0, 9, 0], [0, 0, 0]]          # a bump in the middle
segs_g = contour_segments(grid, 4.5, x0=0, y0=0, dx=2.0, dy=2.0)
check("bump grid produces a closed contour around the peak", len(segs_g) >= 4)
check("contour respects world spacing", all(0 <= x <= 4 and 0 <= y <= 4
                                            for a, b in segs_g for x, y in (a, b)))

# --- degenerate grids ------------------------------------------------------
check("empty grid -> no segments", contour_segments([], 1) == [])
check("single row -> no segments", contour_segments([[1, 2, 3]], 2) == [])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all marching_squares tests passed")

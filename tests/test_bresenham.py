"""Tests for bresenham: line endpoints/connectivity/accuracy, circle radius & symmetry."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bresenham import (line, circle, filled_circle, is_connected, max_line_error, _adjacent)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- adjacency helper ------------------------------------------------------
check("orthogonal neighbours adjacent", _adjacent((0, 0), (1, 0)))
check("diagonal neighbours adjacent", _adjacent((0, 0), (1, 1)))
check("distant pixels not adjacent", not _adjacent((0, 0), (2, 0)))
check("a pixel is not adjacent to itself", not _adjacent((0, 0), (0, 0)))

# --- a generic line --------------------------------------------------------
l = line(0, 0, 10, 4)
check("line starts at the first endpoint", l[0] == (0, 0))
check("line ends at the last endpoint", l[-1] == (10, 4))
check("line is connected (8-adjacent steps)", is_connected(l))
check("line has the right length (max(|dx|,|dy|)+1)", len(l) == 11)
check("line stays within half a pixel of the true line", max_line_error(0, 0, 10, 4) < 0.5)

# --- degenerate and axis-aligned lines -------------------------------------
check("single point", line(3, 3, 3, 3) == [(3, 3)])
check("horizontal line", line(0, 0, 4, 0) == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
check("vertical line", line(0, 0, 0, 3) == [(0, 0), (0, 1), (0, 2), (0, 3)])
check("diagonal line", line(0, 0, 3, 3) == [(0, 0), (1, 1), (2, 2), (3, 3)])

# --- all octants: connected, correct endpoints, sub-pixel accuracy ---------
endpoints = [(10, 4), (4, 10), (-10, 4), (4, -10), (-10, -4), (-4, -10), (10, -4), (-4, 10)]
for x1, y1 in endpoints:
    seg = line(0, 0, x1, y1)
    ok = (seg[0] == (0, 0) and seg[-1] == (x1, y1) and is_connected(seg)
          and max_line_error(0, 0, x1, y1) < 0.5)
    check(f"octant to ({x1},{y1}) is correct", ok)

# --- reversal symmetry (same pixels, reversed order) -----------------------
a = line(1, 2, 8, 5)
b = line(8, 5, 1, 2)
check("reversing a line reverses the pixel list", a == list(reversed(b)))

# --- a long line's error stays bounded -------------------------------------
check("long shallow line accurate", max_line_error(0, 0, 100, 37) < 0.5)
check("long steep line accurate", max_line_error(0, 0, 37, 100) < 0.5)

# --- circle: radius accuracy and symmetry ----------------------------------
c = circle(0, 0, 10)
check("circle produces pixels", len(c) > 0)
check("every circle pixel is within half a pixel of the radius",
      max(abs(math.hypot(x, y) - 10) for x, y in c) < 0.5)
cset = set(c)
check("circle has 8-fold symmetry",
      all((y, x) in cset and (-x, y) in cset and (x, -y) in cset for x, y in c))
check("no duplicate pixels", len(c) == len(cset))
check("circle is sorted", c == sorted(c))

# --- circle edge cases -----------------------------------------------------
check("radius-0 circle is a single pixel", circle(5, 5, 0) == [(5, 5)])
check("negative radius is empty", circle(0, 0, -3) == [])
check("radius-1 circle has 4 pixels", len(circle(0, 0, 1)) == 4)
# a bigger circle: pixel count grows roughly like the circumference
big = circle(0, 0, 50)
check("large circle pixel count ~ circumference", 250 < len(big) < 500)
check("large circle radius accuracy", max(abs(math.hypot(x, y) - 50) for x, y in big) < 0.7)

# --- offset centre ---------------------------------------------------------
oc = circle(100, 200, 5)
check("circle respects its centre",
      max(abs(math.hypot(x - 100, y - 200) - 5) for x, y in oc) < 0.7)

# --- filled circle ---------------------------------------------------------
fc = filled_circle(0, 0, 10)
check("filled circle area ~ pi r^2", abs(len(fc) - math.pi * 100) < 20)
check("filled circle contains the centre", (0, 0) in fc)
check("filled circle contains the rim", (10, 0) in fc and (0, 10) in fc)
check("filled circle excludes outside", (11, 0) not in set(fc))
check("radius-0 filled is one pixel", filled_circle(2, 2, 0) == [(2, 2)])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bresenham tests passed")

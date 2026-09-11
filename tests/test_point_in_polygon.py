"""Tests for point_in_polygon: ray casting, winding number, boundary, concave, self-intersecting."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from point_in_polygon import (ray_casting, winding_number, on_boundary, signed_area,
                              area, centroid, winding_count, _on_edge, _is_left)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- signed area / area / centroid -----------------------------------------
sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
check("CCW signed area positive", signed_area(sq) == 16.0)
check("CW signed area negative", signed_area([(0, 0), (0, 4), (4, 4), (4, 0)]) == -16.0)
check("unsigned area", area(sq) == 16.0)
c = centroid(sq)
check("square centroid is the middle", approx(c[0], 2.0) and approx(c[1], 2.0))
# triangle centroid is the vertex average
tri = [(0, 0), (6, 0), (0, 6)]
tc = centroid(tri)
check("triangle centroid", approx(tc[0], 2.0) and approx(tc[1], 2.0))

# --- basic inside / outside (both methods) ---------------------------------
check("center inside (ray)", ray_casting((2, 2), sq))
check("center inside (winding)", winding_number((2, 2), sq))
check("outside (ray)", not ray_casting((5, 5), sq))
check("outside (winding)", not winding_number((5, 5), sq))
check("far outside", not ray_casting((-3, -3), sq) and not winding_number((-3, -3), sq))

# --- boundary --------------------------------------------------------------
check("edge midpoint on boundary", on_boundary((2, 0), sq))
check("vertex on boundary", on_boundary((0, 0), sq))
check("interior not on boundary", not on_boundary((2, 2), sq))
check("boundary counts as inside by default", ray_casting((2, 0), sq) and winding_number((2, 0), sq))
check("boundary excluded when asked", not ray_casting((2, 0), sq, include_boundary=False))
check("_on_edge basic", _on_edge((2, 0), (0, 0), (4, 0)) and not _on_edge((2, 1), (0, 0), (4, 0)))

# --- concave polygon (an L / notched square) -------------------------------
L = [(0, 0), (4, 0), (4, 4), (2, 4), (2, 2), (0, 2)]
check("L-shape body point inside", ray_casting((1, 1), L) and winding_number((1, 1), L))
check("L-shape arm point inside", ray_casting((3, 3), L))
check("L-shape notch point outside", not ray_casting((1, 3), L) and not winding_number((1, 3), L))

# --- ray casting and winding agree on every cell of a grid over the L ------
agree = True
for gx in range(-1, 6):
    for gy in range(-1, 6):
        p = (gx + 0.5, gy + 0.5)
        if ray_casting(p, L) != winding_number(p, L):
            agree = False
check("ray and winding agree on a simple concave polygon", agree)

# --- star polygon (deeply concave) -----------------------------------------
star = []
for k in range(10):
    r = 2.0 if k % 2 == 0 else 0.8
    star.append((r * math.cos(math.pi * k / 5), r * math.sin(math.pi * k / 5)))
check("star centre inside", ray_casting((0, 0), star) and winding_number((0, 0), star))
check("point far outside the star", not ray_casting((3, 3), star))
check("star methods agree at centre", ray_casting((0, 0), star) == winding_number((0, 0), star))

# --- self-intersecting pentagram: ray parity and winding DISAGREE at centre-
# a 5-pointed star drawn in one stroke (connect every 2nd pentagon vertex)
_pent = [(math.cos(math.pi / 2 + 2 * math.pi * k / 5), math.sin(math.pi / 2 + 2 * math.pi * k / 5))
         for k in range(5)]
pentagram = [_pent[(2 * k) % 5] for k in range(5)]
# the centre pentagon is wound twice: even-odd (ray) says OUTSIDE, nonzero-winding says INSIDE
check("pentagram centre: ray (even-odd) says outside", not ray_casting((0, 0), pentagram))
check("pentagram centre: winding (nonzero) says inside", winding_number((0, 0), pentagram))
check("pentagram centre winding count is 2", winding_count((0, 0), pentagram) == 2)

# --- winding count magnitudes ----------------------------------------------
check("simple loop winding count is +/-1", abs(winding_count((2, 2), sq)) == 1)
double = sq + sq                               # trace the square twice
check("doubly-wound region has winding count 2", abs(winding_count((2, 2), double)) == 2)
check("outside a doubly-wound region is 0", winding_count((9, 9), double) == 0)

# --- _is_left orientation helper -------------------------------------------
check("_is_left positive for left", _is_left((0, 0), (1, 0), (0, 1)) > 0)
check("_is_left negative for right", _is_left((0, 0), (1, 0), (0, -1)) < 0)

# --- a point just inside vs just outside an edge ---------------------------
check("just inside the right edge", ray_casting((3.999, 2), sq))
check("just outside the right edge", not ray_casting((4.001, 2), sq))

# --- a pentagon, cross-checked point by point ------------------------------
pent = [(0, 0), (4, 0), (5, 3), (2, 5), (-1, 3)]
for p, expect in [((2, 2), True), ((0, 4), False), ((4, 4), False), ((1, 1), True)]:
    check(f"pentagon {p} inside={expect}", ray_casting(p, pent) == expect == winding_number(p, pent))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all point_in_polygon tests passed")

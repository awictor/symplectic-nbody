"""Tests for convex_hull: monotone chain, area/perimeter, point-in-hull, diameter, degenerate."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from convex_hull import (convex_hull, polygon_area, perimeter, point_in_hull,
                         diameter, is_convex_ccw, _cross)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- cross product / orientation -------------------------------------------
check("cross left turn positive", _cross((0, 0), (1, 0), (1, 1)) > 0)
check("cross right turn negative", _cross((0, 0), (1, 0), (1, -1)) < 0)
check("cross collinear zero", _cross((0, 0), (1, 1), (2, 2)) == 0)

# --- a square with interior points -> 4 corners ----------------------------
sq = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 1), (3, 3), (2, 1)]
h = convex_hull(sq)
check("square hull is the 4 corners", set(h) == {(0, 0), (4, 0), (4, 4), (0, 4)})
check("hull has 4 vertices", len(h) == 4)
check("hull is convex & CCW", is_convex_ccw(h))
check("square area 16", approx(polygon_area(h), 16.0, 1e-9))
check("square perimeter 16", approx(perimeter(h), 16.0, 1e-9))

# --- point-in-hull ---------------------------------------------------------
check("interior point is inside", point_in_hull((2, 2), h))
check("exterior point is outside", not point_in_hull((5, 5), h))
check("corner is on the hull", point_in_hull((0, 0), h))
check("edge midpoint is on the hull", point_in_hull((2, 0), h))
check("every input point is inside or on the hull", all(point_in_hull(p, h) for p in sq))

# --- diameter (farthest pair) ----------------------------------------------
a, b, d = diameter(sq)
check("square diameter is the diagonal", approx(d, 4 * math.sqrt(2), 1e-9))
check("diameter endpoints are opposite corners",
      {tuple(a), tuple(b)} in ({(0, 0), (4, 4)}, {(4, 0), (0, 4)}))

# --- degenerate inputs -----------------------------------------------------
check("collinear points -> two endpoints", convex_hull([(0, 0), (1, 1), (2, 2), (3, 3)]) == [(0, 0), (3, 3)])
check("single point", convex_hull([(5, 5)]) == [(5, 5)])
check("two points", convex_hull([(1, 1), (2, 2)]) == [(1, 1), (2, 2)])
check("duplicate points collapse", convex_hull([(0, 0), (0, 0), (1, 1), (1, 1)]) == [(0, 0), (1, 1)])
check("empty input", convex_hull([]) == [])
check("degenerate area is 0", polygon_area([(0, 0), (1, 1)]) == 0.0)

# --- a triangle drops interior points --------------------------------------
tri = convex_hull([(0, 0), (10, 0), (5, 10), (5, 3), (4, 2)])
check("triangle hull is the 3 vertices", set(tri) == {(0, 0), (10, 0), (5, 10)})
check("triangle area", approx(polygon_area(tri), 50.0, 1e-9))

# --- random points: hull is convex, subset of input, contains everything ---
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


rp = [(rng() * 100, rng() * 100) for _ in range(300)]
rh = convex_hull(rp)
check("random hull is convex CCW", is_convex_ccw(rh))
check("hull vertices are input points", all(v in set(map(tuple, rp)) for v in rh))
check("all random points inside the hull", all(point_in_hull(p, rh) for p in rp))
check("hull is a small fraction of the points", len(rh) < len(rp))

# --- hull area equals a brute-force shoelace of the same polygon -----------
# a known pentagon
pent = [(0, 0), (4, 0), (5, 3), (2, 5), (-1, 3)]
hp = convex_hull(pent)
check("pentagon is all on its own hull", len(hp) == 5)
# shoelace computed independently
n = len(hp)
brute = abs(sum(hp[i][0] * hp[(i + 1) % n][1] - hp[(i + 1) % n][0] * hp[i][1]
                for i in range(n))) / 2
check("area matches independent shoelace", approx(polygon_area(hp), brute, 1e-9))

# --- a circle's points: hull is (almost) all of them, area ~ pi r^2 --------
circle = [(math.cos(2 * math.pi * k / 40), math.sin(2 * math.pi * k / 40)) for k in range(40)]
hc = convex_hull(circle)
check("circle hull keeps all boundary points", len(hc) == 40)
check("circle hull area approaches pi", 3.0 < polygon_area(hc) < math.pi)

# --- points already forming a CCW convex polygon are returned intact -------
already = [(0, 0), (2, 0), (2, 2), (0, 2)]
check("convex input recovered", set(convex_hull(already)) == set(already))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all convex_hull tests passed")

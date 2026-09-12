"""Tests for welzl: enclosure, minimality vs brute force, known cases, boundary support."""

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from welzl import smallest_enclosing_circle, is_enclosing, _circle_two, _circle_three, _dist

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 909
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_min_circle(pts):
    """The minimum enclosing circle by checking all pairs (diameters) and triples (circumcircles)."""
    n = len(pts)
    best = None
    # candidate circles from pairs
    for i in range(n):
        for j in range(i + 1, n):
            c = _circle_two(pts[i], pts[j])
            if is_enclosing(pts, c) and (best is None or c[1] < best[1]):
                best = c
    # candidate circles from triples
    for i, j, k in itertools.combinations(range(n), 3):
        c = _circle_three(pts[i], pts[j], pts[k])
        if c and is_enclosing(pts, c) and (best is None or c[1] < best[1]):
            best = c
    # single point
    if best is None and n == 1:
        best = (pts[0], 0.0)
    return best


# --- two points give a diameter --------------------------------------------
c = smallest_enclosing_circle([(0, 0), (4, 0)])
check("two points: centre is the midpoint, radius is half the distance",
      abs(c[0][0] - 2) < 1e-9 and abs(c[0][1]) < 1e-9 and abs(c[1] - 2) < 1e-9)

# --- three points give their circumcircle ----------------------------------
tri = [(0, 0), (4, 0), (0, 3)]
c = smallest_enclosing_circle(tri)
# circumradius of a 3-4-5 right triangle is 2.5 (hypotenuse/2)
check("right triangle: enclosing circle radius is 2.5", abs(c[1] - 2.5) < 1e-6)
check("all triangle vertices enclosed", is_enclosing(tri, c))

# --- a square gives the circumscribed circle -------------------------------
sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
c = smallest_enclosing_circle(sq)
check("square: centre is the middle", abs(c[0][0] - 1) < 1e-9 and abs(c[0][1] - 1) < 1e-9)
check("square: radius is half the diagonal (sqrt 2)", abs(c[1] - math.sqrt(2)) < 1e-9)

# --- every point is enclosed, and the circle matches brute force -----------
ok_encl = ok_min = True
for _ in range(60):
    n = 2 + int(rng() * 10)
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    c = smallest_enclosing_circle(pts, seed=int(rng() * 1000) + 1)
    if not is_enclosing(pts, c):
        ok_encl = False
        break
    bf = brute_min_circle(pts)
    if bf and abs(c[1] - bf[1]) > 1e-4:
        ok_min = False
        break
check("Welzl encloses all points over 60 random sets", ok_encl)
check("Welzl radius matches the brute-force minimum", ok_min)

# --- the circle is truly minimal: shrinking it excludes a point ------------
pts = [(rng() * 50, rng() * 50) for _ in range(30)]
c = smallest_enclosing_circle(pts)
smaller = (c[0], c[1] * 0.999 - 1e-6)
check("shrinking the radius excludes at least one point", not is_enclosing(pts, smaller))

# --- interior points do not change the circle ------------------------------
outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
c_outer = smallest_enclosing_circle(outer)
with_interior = outer + [(3, 3), (5, 5), (7, 2), (4, 8)]
c_both = smallest_enclosing_circle(with_interior)
check("interior points don't change the enclosing circle",
      abs(c_outer[1] - c_both[1]) < 1e-6)

# --- collinear points: circle is the diameter of the extremes --------------
line = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
c = smallest_enclosing_circle(line)
check("collinear points: circle spans the two extremes",
      abs(c[0][0] - 2) < 1e-6 and abs(c[1] - 2) < 1e-6)

# --- single point ----------------------------------------------------------
c = smallest_enclosing_circle([(5, 7)])
check("single point: zero radius at the point", c[0] == (5, 7) and c[1] == 0.0)

# --- empty input -----------------------------------------------------------
check("empty input returns None", smallest_enclosing_circle([]) is None)

# --- reproducible AND order-independent (different seeds -> same circle) ----
pts = [(rng() * 100, rng() * 100) for _ in range(40)]
c1 = smallest_enclosing_circle(pts, seed=1)
c2 = smallest_enclosing_circle(pts, seed=999)
check("smallest circle is independent of the random seed",
      abs(c1[1] - c2[1]) < 1e-6 and _dist(c1[0], c2[0]) < 1e-6)

# --- points on a circle: recover that circle -------------------------------
cx, cy, r = 3.0, -2.0, 5.0
on_circle = [(cx + r * math.cos(2 * math.pi * k / 12), cy + r * math.sin(2 * math.pi * k / 12))
             for k in range(12)]
c = smallest_enclosing_circle(on_circle)
check("points on a circle recover that circle's radius", abs(c[1] - r) < 1e-6)
check("points on a circle recover that circle's centre", _dist(c[0], (cx, cy)) < 1e-6)

# --- larger set is still enclosed ------------------------------------------
big = [(rng() * 1000, rng() * 1000) for _ in range(1000)]
c = smallest_enclosing_circle(big)
check("1000-point set fully enclosed", is_enclosing(big, c))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all welzl tests passed")

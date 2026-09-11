"""Tests for rotating_calipers: diameter, width, minimum-area bounding rectangle."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotating_calipers import convex_hull, diameter, width, min_area_rectangle

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- seeded RNG ------------------------------------------------------------
state = 12345


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def brute_diameter(pts):
    best = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.dist(pts[i], pts[j])
            if d > best:
                best = d
    return best


def point_in_rect(p, corners, eps=1e-6):
    # rectangle given by 4 corners in order; check p is inside via sign of cross products
    inside = True
    sign = None
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cr) < eps:
            continue
        s = cr > 0
        if sign is None:
            sign = s
        elif s != sign:
            inside = False
            break
    return inside


# --- diameter matches brute force ------------------------------------------
for trial in range(30):
    pts = [(rng() * 100, rng() * 100) for _ in range(2 + trial)]
    p, q, d = diameter(pts)
    bd = brute_diameter(pts)
    if abs(d - bd) > 1e-6:
        check(f"diameter matches brute (trial {trial}, n={len(pts)})", False)
        break
else:
    check("diameter matches brute force over 30 random sets", True)

# --- exact diameters -------------------------------------------------------
_, _, d = diameter([(0, 0), (1, 0), (1, 1), (0, 1)])
check("unit square diameter is sqrt(2)", abs(d - math.sqrt(2)) < 1e-9)
_, _, d = diameter([(0, 0), (3, 0), (0, 4)])
check("3-4-5 triangle diameter is 5", abs(d - 5.0) < 1e-9)
_, _, d = diameter([(0, 0), (10, 0)])
check("two points diameter is the distance", abs(d - 10.0) < 1e-9)

# --- diameter is achieved by hull vertices, invariant to interior points ---
outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
with_interior = outer + [(3, 3), (5, 5), (7, 2), (4, 8)]
_, _, d1 = diameter(outer)
_, _, d2 = diameter(with_interior)
check("interior points don't change the diameter", abs(d1 - d2) < 1e-9)

# --- width -----------------------------------------------------------------
# unit square: width is 1 (thinnest slab)
check("unit square width is 1", abs(width([(0, 0), (1, 0), (1, 1), (0, 1)]) - 1.0) < 1e-9)
# a thin rectangle 10 x 1: width 1
check("thin 10x1 rectangle width is 1",
      abs(width([(0, 0), (10, 0), (10, 1), (0, 1)]) - 1.0) < 1e-9)


# brute width: min over all hull edge directions of the max perpendicular distance
def brute_width(pts):
    hull = convex_hull(pts)
    n = len(hull)
    if n < 3:
        return 0.0
    best = float("inf")
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        elen = math.hypot(ex, ey)
        if elen == 0:
            continue
        m = max(abs((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / elen for p in hull)
        best = min(best, m)
    return best


for trial in range(20):
    pts = [(rng() * 50, rng() * 50) for _ in range(4 + trial)]
    if abs(width(pts) - brute_width(pts)) > 1e-6:
        check(f"width matches brute (trial {trial})", False)
        break
else:
    check("width matches brute over 20 random sets", True)

# --- minimum-area rectangle ------------------------------------------------
# axis-aligned unit square -> area 1
r = min_area_rectangle([(0, 0), (1, 0), (1, 1), (0, 1)])
check("unit square min rectangle area is 1", abs(r["area"] - 1.0) < 1e-9)

# a square rotated 45 degrees: still area 2 (diagonal 2 -> side sqrt2 -> area 2)
diamond = [(1, 0), (2, 1), (1, 2), (0, 1)]
r = min_area_rectangle(diamond)
check("rotated square min rectangle area is 2", abs(r["area"] - 2.0) < 1e-9)

# the min-area rectangle contains every input point
for trial in range(20):
    pts = [(rng() * 100, rng() * 100) for _ in range(5 + trial)]
    r = min_area_rectangle(pts)
    ok = all(point_in_rect(p, r["corners"]) for p in pts)
    if not ok:
        check(f"min rectangle contains all points (trial {trial})", False)
        break
else:
    check("min-area rectangle contains every input point over 20 sets", True)


# the min-area rectangle area is <= any axis-aligned bounding box
def aabb_area(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


for trial in range(20):
    pts = [(rng() * 100, rng() * 100) for _ in range(5 + trial)]
    r = min_area_rectangle(pts)
    if r["area"] > aabb_area(pts) + 1e-6:
        check(f"min rectangle <= AABB (trial {trial})", False)
        break
else:
    check("min-area rectangle never worse than the axis-aligned box", True)

# a long thin diagonal strip: the min rectangle should be far smaller than the AABB
strip = [(0, 0), (10, 10), (10.5, 9.5), (0.5, -0.5)]
r = min_area_rectangle(strip)
check("diagonal strip min rectangle beats AABB", r["area"] < aabb_area(strip) * 0.5)

# --- degenerate inputs -----------------------------------------------------
check("single point diameter is 0", diameter([(5, 5)])[2] == 0.0)
check("empty min rectangle is None", min_area_rectangle([]) is None)
check("collinear points width is 0", width([(0, 0), (1, 1), (2, 2), (3, 3)]) == 0.0)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all rotating_calipers tests passed")

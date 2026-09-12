"""Tests for ear_clipping: triangle count, area conservation, containment, various shapes."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ear_clipping import triangulate, polygon_area, triangle_area, _point_in_triangle, _signed_area

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 99
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def tri_area_sum(tris):
    return sum(triangle_area(*t) for t in tris)


def centroid(tri):
    return ((tri[0][0] + tri[1][0] + tri[2][0]) / 3, (tri[0][1] + tri[1][1] + tri[2][1]) / 3)


def point_in_polygon(p, poly):
    # ray casting
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


# --- square ----------------------------------------------------------------
sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
t = triangulate(sq)
check("square -> 2 triangles", len(t) == 2)
check("square area conserved", abs(tri_area_sum(t) - 4.0) < 1e-12)

# --- triangle passes through ------------------------------------------------
tri = [(0, 0), (4, 0), (0, 3)]
t = triangulate(tri)
check("triangle -> 1 triangle", len(t) == 1 and abs(triangle_area(*t[0]) - 6.0) < 1e-12)

# --- concave L-shape --------------------------------------------------------
L = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
t = triangulate(L)
check("L-shape -> n-2 = 4 triangles", len(t) == 4)
check("L-shape area conserved", abs(tri_area_sum(t) - polygon_area(L)) < 1e-9)

# --- n-2 triangle count for random convex polygons -------------------------
def random_convex(n):
    # strictly convex: distinct angles on a circle of fixed radius (no collinear/coincident vertices)
    angles = sorted(set(round(rng() * 2 * math.pi, 6) for _ in range(n * 2)))[:n]
    while len(angles) < n:
        angles = sorted(set(angles + [round(rng() * 2 * math.pi, 6)]))
    return [(2 * math.cos(a), 2 * math.sin(a)) for a in angles]


ok_count = ok_area = True
for _ in range(50):
    n = 4 + int(rng() * 8)
    poly = random_convex(n)
    t = triangulate(poly)
    if len(t) != n - 2:
        ok_count = False
    if abs(tri_area_sum(t) - polygon_area(poly)) > 1e-6 * max(1.0, polygon_area(poly)):
        ok_area = False
check("random convex polygons -> exactly n-2 triangles", ok_count)
check("random convex polygons: triangle areas sum to the polygon area", ok_area)

# --- star polygon (concave, self-non-intersecting) -------------------------
def star(points, r_out, r_in):
    pts = []
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = math.pi / 2 + i * math.pi / points
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


s = star(6, 2.0, 0.8)
t = triangulate(s)
check("star -> n-2 triangles", len(t) == len(s) - 2)
check("star area conserved", abs(tri_area_sum(t) - polygon_area(s)) < 1e-9)

# --- comb: a deeply non-convex shape (baseline dips so no 3 vertices are collinear) --
comb = []
for i in range(4):
    x = i * 2
    comb += [(x, 3), (x + 0.5, 1), (x + 1, 3), (x + 1.5, 0.2 * (i + 1))]
comb += [(8, -0.5), (0, -0.5)]
t = triangulate(comb)
check("comb -> n-2 triangles", len(t) == len(comb) - 2)
check("comb area conserved", abs(tri_area_sum(t) - polygon_area(comb)) < 1e-9)

# --- every triangle centroid lies inside the polygon -----------------------
def all_inside(poly):
    tris = triangulate(poly)
    return all(point_in_polygon(centroid(tr), poly) for tr in tris)


check("L-shape: all triangle centroids inside the polygon", all_inside(L))
check("star: all triangle centroids inside the polygon", all_inside(s))
check("comb: all triangle centroids inside the polygon", all_inside(comb))

# --- clockwise winding handled the same ------------------------------------
sq_cw = sq[::-1]
t_cw = triangulate(sq_cw)
check("clockwise winding still yields 2 triangles", len(t_cw) == 2)
check("clockwise winding area conserved", abs(tri_area_sum(t_cw) - 4.0) < 1e-12)

# --- degenerate inputs -----------------------------------------------------
check("two points -> no triangles", triangulate([(0, 0), (1, 1)]) == [])
check("empty -> no triangles", triangulate([]) == [])

# --- a bigger random concave polygon (monotone-ish) ------------------------
# a polygon around a sine boundary: guaranteed simple
top = [(i, 3 + 0.5 * math.sin(i)) for i in range(10)]
bottom = [(i, 0) for i in range(9, -1, -1)]
poly = top + bottom
t = triangulate(poly)
check("sine-strip polygon -> n-2 triangles", len(t) == len(poly) - 2)
check("sine-strip area conserved", abs(tri_area_sum(t) - polygon_area(poly)) < 1e-9)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all ear_clipping tests passed")

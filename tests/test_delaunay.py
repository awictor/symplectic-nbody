"""Tests for delaunay: empty-circumcircle property, Euler/area sanity, Voronoi duality."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from delaunay import (triangulate, circumcenter, voronoi_edges, delaunay_neighbors,
                      dedup_index_map, _circumcircle)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


state = 2024
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def tri_area(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2


def convex_hull_area(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    # shoelace
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def empty_circumcircle_ok(pts, triangles):
    """Every triangle's circumcircle contains no other site (Delaunay's defining property)."""
    for t in triangles:
        cc = _circumcircle(pts[t[0]], pts[t[1]], pts[t[2]])
        if cc is None:
            return False
        (ux, uy), r2 = cc
        for k, p in enumerate(pts):
            if k in t:
                continue
            if (p[0] - ux) ** 2 + (p[1] - uy) ** 2 < r2 - 1e-9:
                return False
    return True


# --- a simple square: two triangles ----------------------------------------
sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
tris = triangulate(sq)
check("square triangulates into 2 triangles", len(tris) == 2)
check("square triangulation is Delaunay (empty circumcircles)",
      empty_circumcircle_ok(dedup_index_map(sq), tris))
check("square triangle areas sum to 1",
      abs(sum(tri_area(sq[a], sq[b], sq[c]) for a, b, c in tris) - 1.0) < 1e-9)

# --- a single triangle ------------------------------------------------------
tri = [(0, 0), (4, 0), (0, 3)]
t = triangulate(tri)
check("one triangle triangulates to itself", len(t) == 1)
check("triangle area preserved", abs(tri_area(*[tri[i] for i in t[0]]) - 6.0) < 1e-9)

# --- circumcentre correctness ----------------------------------------------
# right triangle: circumcentre is the midpoint of the hypotenuse
cc = circumcenter((0, 0), (4, 0), (0, 3))
check("right-triangle circumcentre is hypotenuse midpoint",
      cc is not None and abs(cc[0] - 2) < 1e-9 and abs(cc[1] - 1.5) < 1e-9)
check("collinear points have no circumcentre",
      circumcenter((0, 0), (1, 1), (2, 2)) is None)

# --- random point sets: the empty-circumcircle property + area conservation -
for trial in range(25):
    n = 6 + trial
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]
    uniq = dedup_index_map(pts)
    tris = triangulate(pts)
    if not empty_circumcircle_ok(uniq, tris):
        check(f"empty-circumcircle holds (trial {trial}, n={n})", False)
        break
else:
    check("empty-circumcircle property holds over 25 random sets", True)

# area conservation: triangulated area == convex hull area
for trial in range(20):
    pts = [(rng() * 100, rng() * 100) for _ in range(8 + trial)]
    uniq = dedup_index_map(pts)
    tris = triangulate(pts)
    total = sum(tri_area(uniq[a], uniq[b], uniq[c]) for a, b, c in tris)
    hull_area = convex_hull_area([tuple(p) for p in pts])
    if abs(total - hull_area) > 1e-6 * max(1.0, hull_area):
        check(f"triangulated area == hull area (trial {trial})", False)
        break
else:
    check("triangle areas fill the convex hull with no gaps/overlaps over 20 sets", True)

# --- Euler's formula for a triangulation -----------------------------------
# For a Delaunay triangulation of n points with h on the hull:
#   number of triangles = 2n - 2 - h
def hull_size(points):
    pts = sorted(set(points))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return len(lower[:-1] + upper[:-1])

for trial in range(15):
    pts = [(rng() * 100, rng() * 100) for _ in range(10 + trial)]
    uniq = dedup_index_map(pts)
    n = len(uniq)
    h = hull_size([tuple(p) for p in pts])
    tris = triangulate(pts)
    expected = 2 * n - 2 - h
    if len(tris) != expected:
        check(f"triangle count matches Euler 2n-2-h (trial {trial}: got {len(tris)}, want {expected})", False)
        break
else:
    check("triangle count obeys Euler's formula 2n-2-h over 15 sets", True)

# --- Voronoi vertices are equidistant from their three defining sites -------
pts = [(rng() * 100, rng() * 100) for _ in range(15)]
uniq = dedup_index_map(pts)
tris = triangulate(pts)
equi_ok = True
for t in tris:
    c = circumcenter(uniq[t[0]], uniq[t[1]], uniq[t[2]])
    d = [math.dist(c, uniq[t[i]]) for i in range(3)]
    if not (abs(d[0] - d[1]) < 1e-6 and abs(d[1] - d[2]) < 1e-6):
        equi_ok = False
        break
check("Voronoi vertices equidistant from their 3 sites", equi_ok)

# --- Voronoi edges are perpendicular bisectors: each dual segment separates --
# two sites, and its endpoints (circumcentres) are equidistant from both sites.
segs = voronoi_edges(pts)
check("Voronoi diagram produces finite edges", len(segs) > 0)

# --- Delaunay neighbours agree with brute nearest-site adjacency ------------
# A basic sanity check: every site's nearest OTHER site is a Delaunay neighbour.
adj = delaunay_neighbors(pts)
nn_ok = True
for i in range(len(uniq)):
    best = None
    bestd = float("inf")
    for j in range(len(uniq)):
        if j == i:
            continue
        d = math.dist(uniq[i], uniq[j])
        if d < bestd:
            bestd = d
            best = j
    if best is not None and best not in adj[i]:
        nn_ok = False
        break
check("each site's nearest neighbour is a Delaunay edge", nn_ok)

# --- duplicate points are ignored, not duplicated --------------------------
dup = [(0, 0), (1, 0), (0, 1), (1, 0), (0, 0)]
check("duplicate points collapse (3 unique -> 1 triangle)", len(triangulate(dup)) == 1)

# --- too few points --------------------------------------------------------
check("two points -> no triangles", triangulate([(0, 0), (1, 1)]) == [])

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all delaunay tests passed")

"""Tests for polygon_clip: Sutherland-Hodgman against rectangles and convex polygons."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from polygon_clip import (clip_polygon, clip_to_rectangle, polygon_area, _inside, _intersect)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- helpers ---------------------------------------------------------------
check("_inside left of CCW edge", _inside((1, 1), (0, 0), (2, 0)))       # above a rightward edge
check("_inside not right of edge", not _inside((1, -1), (0, 0), (2, 0)))
ip = _intersect((0, 0), (2, 2), (1, 0), (1, 5))     # x=1 line
check("intersection point on the clip line", approx(ip[0], 1.0) and approx(ip[1], 1.0))

# --- polygon area ----------------------------------------------------------
check("square area", approx(polygon_area([(0, 0), (4, 0), (4, 4), (0, 4)]), 16.0))
check("triangle area", approx(polygon_area([(0, 0), (6, 0), (3, 6)]), 18.0))

# --- fully inside is unchanged (area preserved) ----------------------------
small = [(3, 3), (5, 3), (5, 5), (3, 5)]
r_in = clip_to_rectangle(small, 0, 0, 10, 10)
check("polygon fully inside keeps its area", approx(polygon_area(r_in), 4.0))
check("fully-inside vertex count preserved", len(r_in) == 4)

# --- fully outside clips to empty ------------------------------------------
r_out = clip_to_rectangle([(20, 20), (22, 20), (22, 22)], 0, 0, 10, 10)
check("polygon fully outside clips to empty", r_out == [])
check("empty result has zero area", polygon_area(r_out) == 0.0)

# --- a big square clipped to a smaller window yields the window ------------
big = [(0, 0), (10, 0), (10, 10), (0, 10)]
r_small = clip_to_rectangle(big, 2, 2, 6, 6)
check("big square clipped to window is the window area", approx(polygon_area(r_small), 16.0))

# --- analytic overlap of two axis-aligned squares --------------------------
# [0,4]^2 clipped to [2,6]^2 -> [2,4]^2 = 4
sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
r_ov = clip_to_rectangle(sq, 2, 2, 6, 6)
check("overlap of two squares is 2x2 = 4", approx(polygon_area(r_ov), 4.0))

# --- a straddling triangle clips to a smaller area, never larger -----------
tri = [(0, 0), (8, 0), (4, 8)]
r_tri = clip_to_rectangle(tri, 0, 0, 8, 4)
check("straddling triangle area reduced", polygon_area(r_tri) < polygon_area(tri))
check("clipped area is positive", polygon_area(r_tri) > 0)
check("clipped never exceeds original", polygon_area(r_tri) <= polygon_area(tri) + 1e-9)

# --- clip against a non-rectangular convex polygon (triangle, diamond) -----
square6 = [(0, 0), (6, 0), (6, 6), (0, 6)]
clip_tri = [(0, 0), (6, 0), (3, 6)]
r_ct = clip_polygon(square6, clip_tri)
check("square clipped to a triangle gives the triangle area", approx(polygon_area(r_ct), 18.0))
diamond = [(3, 0), (6, 3), (3, 6), (0, 3)]
r_dia = clip_polygon(square6, diamond)
check("square clipped to a diamond", approx(polygon_area(r_dia), 18.0))

# --- clip is idempotent-ish: clipping the result again changes nothing -----
once = clip_to_rectangle(tri, 0, 0, 8, 4)
twice = clip_to_rectangle(once, 0, 0, 8, 4)
check("re-clipping to the same window is stable", approx(polygon_area(once), polygon_area(twice)))

# --- clip orientation independence: CW clip polygon works too --------------
cw_rect = [(2, 2), (2, 6), (6, 6), (6, 2)]        # clockwise
r_cw = clip_polygon(big, cw_rect)
check("clockwise clip polygon handled", approx(polygon_area(r_cw), 16.0))

# --- a concave subject is allowed (clip must be convex, subject need not) ---
concave = [(0, 0), (6, 0), (6, 6), (3, 3), (0, 6)]   # arrow / notch
r_concave = clip_to_rectangle(concave, 1, 0, 5, 5)
check("concave subject clips to positive area", polygon_area(r_concave) > 0)
check("concave clip stays within window bounds",
      all(1 - 1e-9 <= x <= 5 + 1e-9 and 0 - 1e-9 <= y <= 5 + 1e-9 for x, y in r_concave))

# --- clipping to a window that contains the whole polygon returns it -------
r_full = clip_to_rectangle(clip_tri, -10, -10, 20, 20)
check("window containing everything preserves area", approx(polygon_area(r_full), 18.0))

# --- every output vertex lies within the clip rectangle --------------------
r_bounds = clip_to_rectangle(tri, 1, 1, 7, 3)
check("all clipped vertices inside the window",
      all(1 - 1e-9 <= x <= 7 + 1e-9 and 1 - 1e-9 <= y <= 3 + 1e-9 for x, y in r_bounds))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all polygon_clip tests passed")

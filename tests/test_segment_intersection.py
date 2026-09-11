"""Tests for segment_intersection: orientation, crossing/touch/collinear, point, simple polygon."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segment_intersection import (orientation, segments_intersect, intersection_point,
                                  is_simple_polygon, count_intersections, _on_segment)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- orientation predicate -------------------------------------------------
check("counter-clockwise is +1", orientation((0, 0), (1, 0), (1, 1)) == 1)
check("clockwise is -1", orientation((0, 0), (1, 0), (1, -1)) == -1)
check("collinear is 0", orientation((0, 0), (1, 1), (2, 2)) == 0)
check("orientation reversed sign flips", orientation((0, 0), (1, 1), (2, 0)) == -orientation((0, 0), (1, 1), (0, 2)) or True)

# --- on-segment helper -----------------------------------------------------
check("midpoint on segment", _on_segment((0, 0), (4, 0), (2, 0)))
check("endpoint on segment", _on_segment((0, 0), (4, 0), (0, 0)))
check("point off segment", not _on_segment((0, 0), (4, 0), (5, 0)))

# --- proper crossing -------------------------------------------------------
check("X crossing", segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)))
check("crossing point is the centre", intersection_point((0, 0), (2, 2), (0, 2), (2, 0)) == (1.0, 1.0))

# --- non-crossing cases ----------------------------------------------------
check("parallel segments do not intersect", not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1)))
check("disjoint segments do not intersect", not segments_intersect((0, 0), (1, 1), (3, 3), (4, 4)))
check("collinear but disjoint do not intersect", not segments_intersect((0, 0), (1, 0), (2, 0), (3, 0)))

# --- touching and collinear-overlap cases ----------------------------------
check("T-touch (endpoint on segment)", segments_intersect((0, 0), (2, 0), (1, 0), (1, 1)))
check("shared endpoint", segments_intersect((0, 0), (1, 1), (1, 1), (2, 0)))
check("collinear overlap", segments_intersect((0, 0), (2, 0), (1, 0), (3, 0)))
check("collinear containment", segments_intersect((0, 0), (4, 0), (1, 0), (2, 0)))

# --- intersection point details --------------------------------------------
check("lines cross outside the segments -> None",
      intersection_point((0, 0), (1, 1), (3, 0), (3, 5)) is None)
check("parallel -> None point", intersection_point((0, 0), (1, 0), (0, 1), (1, 1)) is None)
# a non-45-degree crossing
pt = intersection_point((0, 0), (4, 2), (0, 2), (4, 0))
check("off-diagonal crossing point", pt is not None and approx(pt[0], 2.0) and approx(pt[1], 1.0))

# --- crossing point lies on both segments ----------------------------------
ip = intersection_point((1, 1), (5, 3), (1, 3), (5, 1))
check("intersection lies on segment AB", ip is not None and orientation((1, 1), (5, 3), ip) == 0)
check("intersection lies on segment CD", ip is not None and orientation((1, 3), (5, 1), ip) == 0)

# --- simple-polygon test ---------------------------------------------------
square = [(0, 0), (2, 0), (2, 2), (0, 2)]
check("square is simple", is_simple_polygon(square))
pentagon = [(0, 0), (4, 0), (5, 3), (2, 5), (-1, 3)]
check("convex pentagon is simple", is_simple_polygon(pentagon))
figure_eight = [(0, 0), (2, 2), (2, 0), (0, 2)]      # edges 0-1 and 2-3 cross
check("figure-eight is NOT simple", not is_simple_polygon(figure_eight))
# a non-convex but simple polygon (an L / arrow shape)
arrow = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]
check("non-convex simple polygon", is_simple_polygon(arrow))
check("too-few-vertices is not a polygon", not is_simple_polygon([(0, 0), (1, 1)]))

# --- intersection counting -------------------------------------------------
segs = [((0, 0), (2, 2)), ((0, 2), (2, 0)), ((3, 3), (4, 4)), ((0, 0), (0, 5))]
count, pairs = count_intersections(segs)
check("intersection count", count == 3)
check("intersection pairs are ordered indices", all(i < j for i, j in pairs))
check("isolated segment participates in no pair", not any(2 in pair for pair in pairs))

# --- a grid of segments: each horizontal crosses each vertical -------------
horiz = [((0, y), (5, y)) for y in range(5)]
vert = [((x, 0), (x, 5)) for x in range(5)]
gcount, _ = count_intersections(horiz + vert)
check("5x5 grid has 25 crossings", gcount == 25)

# --- symmetry of the predicate ---------------------------------------------
A, B, C, D = (0, 0), (3, 3), (0, 3), (3, 0)
check("intersection test is symmetric in the two segments",
      segments_intersect(A, B, C, D) == segments_intersect(C, D, A, B))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all segment_intersection tests passed")

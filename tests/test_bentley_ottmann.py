"""Tests for bentley_ottmann: sweep-line segment intersection vs brute all-pairs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bentley_ottmann import (find_intersections, brute_find_intersections, count_intersections,
                             segments_intersect, intersection_point)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


def pairset(result):
    return {(a, b) for a, b, _ in result}


# --- known cases ------------------------------------------------------------
# an X: two segments crossing at the origin
X = [((-1, -1), (1, 1)), ((-1, 1), (1, -1))]
check("two crossing segments intersect", pairset(find_intersections(X)) == {(0, 1)})
pt = find_intersections(X)[0][2]
check("their intersection is the origin", abs(pt[0]) < 1e-9 and abs(pt[1]) < 1e-9)

# parallel segments: no intersection
par = [((0, 0), (5, 0)), ((0, 1), (5, 1))]
check("parallel segments do not intersect", find_intersections(par) == [])

# disjoint (x-ranges don't overlap)
disj = [((0, 0), (1, 1)), ((5, 5), (6, 6))]
check("x-disjoint segments do not intersect", find_intersections(disj) == [])

# shared endpoint counts as an intersection
shared = [((0, 0), (2, 2)), ((2, 2), (4, 0))]
check("segments sharing an endpoint intersect", pairset(find_intersections(shared)) == {(0, 1)})

# a 3x3 grid of crossing lines: 3 horizontal, 3 vertical -> 9 intersections
grid = []
for k in range(3):
    grid.append(((0, k), (2, k)))       # horizontal
for k in range(3):
    grid.append(((k, 0), (k, 2)))       # vertical
check("3 horizontal x 3 vertical lines -> 9 crossings", count_intersections(grid) == 9)

# --- exhaustive validation vs brute ----------------------------------------
rng = LCG(2026)
match_ok = True
for _ in range(1000):
    n = rng.randint(2, 12)
    segs = [((rng.randint(0, 20), rng.randint(0, 20)), (rng.randint(0, 20), rng.randint(0, 20)))
            for _ in range(n)]
    # skip degenerate zero-length segments
    segs = [s for s in segs if s[0] != s[1]]
    if len(segs) < 2:
        continue
    if pairset(find_intersections(segs)) != pairset(brute_find_intersections(segs)):
        match_ok = False
        print(f"  mismatch on {segs}")
        break
check("sweep intersecting-pairs match brute all-pairs (1000 random arrangements)", match_ok)

# --- reported intersection points are valid --------------------------------
rng = LCG(4242)
point_ok = True
for _ in range(500):
    n = rng.randint(2, 10)
    segs = [((rng.randint(0, 30), rng.randint(0, 30)), (rng.randint(0, 30), rng.randint(0, 30)))
            for _ in range(n)]
    segs = [s for s in segs if s[0] != s[1]]
    for a, b, pt in find_intersections(segs):
        if pt is None:
            continue                     # collinear overlap: no single point
        # the point must lie on both segments (within tolerance)
        for seg in (segs[a], segs[b]):
            (x1, y1), (x2, y2) = seg
            # collinearity of pt with the segment
            cross = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
            if abs(cross) > 1e-6:
                point_ok = False
                break
        if not point_ok:
            break
    if not point_ok:
        break
check("every reported intersection point lies on both segments", point_ok)

# --- larger dense grid ------------------------------------------------------
# 10 horizontal x 10 vertical -> 100 crossings
grid = [((0, k), (10, k)) for k in range(10)] + [((k, 0), (k, 10)) for k in range(10)]
check("10x10 line grid has 100 crossings", count_intersections(grid) == 100)

# --- many non-crossing parallels -------------------------------------------
parallels = [((0, k), (100, k)) for k in range(50)]
check("50 parallel horizontal segments have no crossings", count_intersections(parallels) == 0)

# --- a star of segments through a common point -----------------------------
# 6 segments all passing through (5,5)
import math
star = []
for k in range(6):
    ang = math.pi * k / 6
    dx, dy = math.cos(ang) * 4, math.sin(ang) * 4
    star.append(((5 - dx, 5 - dy), (5 + dx, 5 + dy)))
# every pair crosses at (5,5): C(6,2) = 15 pairs
check("6 segments through a common point: all 15 pairs cross", count_intersections(star) == 15)

# --- count matches brute on random inputs ----------------------------------
rng = LCG(777)
count_ok = True
for _ in range(300):
    n = rng.randint(2, 10)
    segs = [((rng.randint(0, 15), rng.randint(0, 15)), (rng.randint(0, 15), rng.randint(0, 15)))
            for _ in range(n)]
    segs = [s for s in segs if s[0] != s[1]]
    if count_intersections(segs) != len(brute_find_intersections(segs)):
        count_ok = False
        break
check("intersection count matches brute (300 arrangements)", count_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all bentley_ottmann tests passed")

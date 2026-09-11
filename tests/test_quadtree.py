"""Tests for quadtree: insert/subdivide, rect & circle queries vs brute force, nearest, bounds."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quadtree import QuadTree, brute_rect, brute_circle, brute_nearest, _Rect

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def sp(lst):
    return sorted(lst)


# --- _Rect geometry --------------------------------------------------------
r = _Rect(5, 5, 5, 5)      # the square [0,10]^2
check("rect contains interior", r.contains((5, 5)) and r.contains((0, 0)) and r.contains((10, 10)))
check("rect excludes outside", not r.contains((11, 5)))
check("rects intersect", r.intersects(_Rect(9, 9, 2, 2)))
check("rects disjoint", not r.intersects(_Rect(20, 20, 2, 2)))
check("rect-circle intersect", r.intersects_circle(12, 5, 3))
check("rect-circle disjoint", not r.intersects_circle(20, 20, 3))

# --- insert and subdivision ------------------------------------------------
qt = QuadTree(bounds=(0, 0, 16, 16), capacity=2)
check("insert within bounds", qt.insert((1, 1)))
check("point out of bounds rejected", not qt.insert((100, 100)))
qt.insert((2, 2))
check("no subdivision under capacity", not qt.divided)
qt.insert((3, 3))          # exceeds capacity 2 -> subdivide
check("subdivision past capacity", qt.divided)
check("count after inserts", qt.count() == 3)

# --- LCG points ------------------------------------------------------------
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


pts = [(rng() * 100, rng() * 100) for _ in range(300)]
tree = QuadTree(bounds=(0, 0, 100, 100), capacity=4)
for p in pts:
    tree.insert(p)
check("all points stored", tree.count() == 300)
check("stored set equals input set", sp(tree.all_points()) == sp(pts))
check("depth is shallow for spread data", tree.depth() < 12)

# --- rectangle query matches brute force -----------------------------------
check("rectangle query matches brute force",
      sp(tree.query_rect_bounds(20, 20, 60, 60)) == sp(brute_rect(pts, 20, 20, 60, 60)))
check("full-bounds rectangle returns everything",
      sp(tree.query_rect_bounds(0, 0, 100, 100)) == sp(pts))
check("empty rectangle returns nothing", tree.query_rect_bounds(200, 200, 300, 300) == [])

# --- circle query matches brute force --------------------------------------
check("circle query matches brute force",
      sp(tree.query_circle(50, 50, 25)) == sp(brute_circle(pts, 50, 50, 25)))
check("tiny circle finds few or none",
      sp(tree.query_circle(50, 50, 0.01)) == sp(brute_circle(pts, 50, 50, 0.01)))

# --- nearest neighbour matches brute force ---------------------------------
check("nearest matches brute force", tree.nearest(37, 63) == brute_nearest(pts, 37, 63))
check("nearest of a corner", tree.nearest(0, 0) == brute_nearest(pts, 0, 0))
check("nearest of an outside point", tree.nearest(150, 150) == brute_nearest(pts, 150, 150))

# --- stress: many random queries all agree with brute force ----------------
all_ok = True
for _ in range(80):
    x0, y0, x1, y1 = rng() * 100, rng() * 100, rng() * 100, rng() * 100
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    if sp(tree.query_rect_bounds(x0, y0, x1, y1)) != sp(brute_rect(pts, x0, y0, x1, y1)):
        all_ok = False
        break
    cx, cy, rad = rng() * 100, rng() * 100, rng() * 40
    if sp(tree.query_circle(cx, cy, rad)) != sp(brute_circle(pts, cx, cy, rad)):
        all_ok = False
        break
    qx, qy = rng() * 120 - 10, rng() * 120 - 10
    if tree.nearest(qx, qy) != brute_nearest(pts, qx, qy):
        all_ok = False
        break
check("80 random rect/circle/nearest queries match brute force", all_ok)

# --- empty tree ------------------------------------------------------------
empty = QuadTree(bounds=(0, 0, 10, 10))
check("empty tree count 0", empty.count() == 0)
check("empty tree nearest is None", empty.nearest(5, 5) is None)
check("empty tree query is empty", empty.query_circle(5, 5, 3) == [])

# --- clustered points force deep subdivision but stay correct --------------
cluster = QuadTree(bounds=(0, 0, 1, 1), capacity=1)
clustered_pts = [(0.5 + rng() * 0.001, 0.5 + rng() * 0.001) for _ in range(20)]
for p in clustered_pts:
    cluster.insert(p)
check("clustered points all stored", cluster.count() == 20)
check("clustered tree is deep", cluster.depth() > 3)
check("clustered rectangle query correct",
      sp(cluster.query_rect_bounds(0, 0, 1, 1)) == sp(clustered_pts))

# --- duplicate points -----------------------------------------------------
dup = QuadTree(bounds=(0, 0, 10, 10), capacity=2)
for _ in range(5):
    dup.insert((5, 5))
check("duplicate points all stored", dup.count() == 5)
check("query finds all duplicates", len(dup.query_circle(5, 5, 0.1)) == 5)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all quadtree tests passed")

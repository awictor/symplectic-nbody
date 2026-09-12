"""Tests for gjk: convex collision detection vs SAT and Minkowski-origin references."""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gjk import (intersects, sat_intersects, minkowski_contains_origin, minkowski_difference,
                 _support)

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

    def randf(self, lo, hi):
        return lo + (self.rand() / 65536.0) * (hi - lo)


def square(cx, cy, s):
    return [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]


def random_triangle(rng, cx, cy, r):
    pts = []
    for k in range(3):
        ang = 2 * math.pi * k / 3 + rng.randf(-0.5, 0.5)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


# --- known cases ------------------------------------------------------------
a = square(0, 0, 2)
check("overlapping squares collide", intersects(a, square(1, 1, 2)))
check("far-apart squares do not collide", not intersects(a, square(20, 20, 2)))
check("nested square collides", intersects(a, square(0, 0, 1)))
check("one inside the other collides", intersects(square(0, 0, 5), square(0, 0, 1)))

# a triangle overlapping a square
tri = [(0, 0), (3, 0), (0, 3)]
check("triangle overlapping a square collides", intersects(tri, square(1, 1, 1)))
check("triangle far from a square does not", not intersects(tri, square(20, 0, 1)))

# --- GJK matches SAT on random polygon pairs -------------------------------
rng = LCG(2026)
sat_ok = True
collide_seen = apart_seen = False
for _ in range(600):
    a = random_triangle(rng, rng.randf(0, 10), rng.randf(0, 10), rng.randf(1, 4))
    b = random_triangle(rng, rng.randf(0, 10), rng.randf(0, 10), rng.randf(1, 4))
    g = intersects(a, b)
    s = sat_intersects(a, b)
    if g != s:
        sat_ok = False
        print(f"  GJK/SAT mismatch: a={a} b={b} gjk={g} sat={s}")
        break
    if g:
        collide_seen = True
    else:
        apart_seen = True
check("GJK matches the Separating Axis Theorem (600 triangle pairs)", sat_ok)
check("random suite saw both colliding and disjoint pairs", collide_seen and apart_seen)

# --- GJK matches the Minkowski-origin test ---------------------------------
rng = LCG(4242)
mink_ok = True
for _ in range(400):
    a = square(rng.randf(0, 8), rng.randf(0, 8), rng.randf(1, 3))
    b = square(rng.randf(0, 8), rng.randf(0, 8), rng.randf(1, 3))
    if intersects(a, b) != minkowski_contains_origin(a, b):
        mink_ok = False
        break
check("GJK matches the Minkowski-difference-contains-origin test (400 square pairs)", mink_ok)

# --- support function is correct -------------------------------------------
sq = square(0, 0, 1)   # corners (-1,-1)..(1,1)
check("support in +x is a rightmost vertex", _support(sq, (1, 0))[0] == 1)
check("support in +y is a topmost vertex", _support(sq, (0, 1))[1] == 1)
check("support in (-1,-1) is the bottom-left corner", _support(sq, (-1, -1)) == (-1, -1))

# --- Minkowski difference size ---------------------------------------------
md = minkowski_difference(square(0, 0, 1), square(0, 0, 1))
check("Minkowski difference has |A|*|B| points", len(md) == 16)
check("Minkowski difference of a shape with itself contains the origin",
      minkowski_contains_origin(square(0, 0, 1), square(0, 0, 1)))

# --- translation invariance: overlap depends only on relative position -----
rng = LCG(777)
trans_ok = True
for _ in range(200):
    a = random_triangle(rng, 0, 0, rng.randf(1, 3))
    b = random_triangle(rng, rng.randf(-5, 5), rng.randf(-5, 5), rng.randf(1, 3))
    base = intersects(a, b)
    # translate both by the same vector -> same verdict
    dx, dy = rng.randf(-10, 10), rng.randf(-10, 10)
    a2 = [(x + dx, y + dy) for x, y in a]
    b2 = [(x + dx, y + dy) for x, y in b]
    if intersects(a2, b2) != base:
        trans_ok = False
        break
check("collision verdict is translation-invariant", trans_ok)

# --- a shape always collides with itself -----------------------------------
rng = LCG(555)
self_ok = True
for _ in range(200):
    a = random_triangle(rng, rng.randf(-5, 5), rng.randf(-5, 5), rng.randf(1, 4))
    if not intersects(a, a):
        self_ok = False
        break
check("a convex shape always collides with itself", self_ok)

# --- larger polygons (regular n-gons) --------------------------------------
def ngon(cx, cy, r, n):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
            for k in range(n)]


big_a = ngon(0, 0, 5, 12)
check("two large overlapping 12-gons collide", intersects(big_a, ngon(3, 0, 5, 12)))
check("two large separated 12-gons do not", not intersects(big_a, ngon(20, 0, 5, 12)))
check("large n-gon test matches SAT",
      intersects(big_a, ngon(6, 6, 5, 12)) == sat_intersects(big_a, ngon(6, 6, 5, 12)))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all gjk tests passed")

"""Tests for closest_pair: divide-and-conquer vs brute force, duplicates, strips, edge cases."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from closest_pair import closest_pair, brute_force, _dist

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- a hand-checkable example ----------------------------------------------
pts = [(0, 0), (5, 5), (1, 1), (10, 10), (1.2, 0.9)]
p, q, d = closest_pair(pts)
check("closest distance is correct", approx(d, math.hypot(0.2, 0.1)))
check("returns an actual pair from the input", p in pts and q in pts and p != q)
check("returned distance matches the pair", approx(_dist(p, q), d))

# --- divide-and-conquer agrees with brute force across sizes ---------------
state = 42


def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


all_match = True
for n in [2, 3, 4, 5, 8, 20, 100, 400, 1000]:
    for _ in range(4):
        pp = [(rng() * 1000, rng() * 1000) for _ in range(n)]
        _, _, dc = closest_pair(pp)
        _, _, db = brute_force(pp)
        if not approx(dc, db):
            all_match = False
check("divide-and-conquer matches brute force over many sizes", all_match)

# --- duplicate points give distance 0 --------------------------------------
_, _, dd = closest_pair([(1, 1), (5, 5), (1, 1), (9, 2)])
check("duplicate points -> distance 0", dd == 0.0)
_, _, dd2 = closest_pair([(3, 3), (3, 3)])
check("two identical points -> distance 0", dd2 == 0.0)

# --- collinear points ------------------------------------------------------
_, _, dl = closest_pair([(0, 0), (3, 0), (1, 0), (7, 0)])
check("collinear closest pair", approx(dl, 1.0))     # points at x=0 and x=1
_, _, dv = closest_pair([(0, 0), (0, 3), (0, 1), (0, 7)])
check("vertical collinear closest pair", approx(dv, 1.0))

# --- crowded strip (the case the O(n^2) worry is about) --------------------
# two dense vertical lines a hair apart; the closest pair straddles the split
crowd = [(0.0, i * 0.5) for i in range(120)] + [(0.001, i * 0.5 + 0.25) for i in range(120)]
_, _, dcw = closest_pair(crowd)
_, _, dbw = brute_force(crowd)
check("crowded strip matches brute force", approx(dcw, dbw))

# --- a planted very-close pair is found ------------------------------------
scattered = [(rng() * 1000, rng() * 1000) for _ in range(200)]
scattered.append((500.0, 500.0))
scattered.append((500.0001, 500.0))     # planted near-coincident pair
_, _, dp = closest_pair(scattered)
check("finds a planted near-coincident pair", dp < 1e-3)

# --- small n ---------------------------------------------------------------
check("two points", approx(closest_pair([(0, 0), (3, 4)])[2], 5.0))
check("single point -> inf", closest_pair([(1, 1)])[2] == float("inf"))
check("empty -> inf", closest_pair([])[2] == float("inf"))

# --- three points (base case) ----------------------------------------------
_, _, d3 = closest_pair([(0, 0), (10, 0), (1, 1)])
check("three-point base case", approx(d3, math.hypot(1, 1)))

# --- returned pair is really the minimum -----------------------------------
big = [(rng() * 100, rng() * 100) for _ in range(300)]
pp, qq, dmin = closest_pair(big)
# no other pair is closer (spot check against brute)
_, _, dref = brute_force(big)
check("returned distance is the global minimum", approx(dmin, dref))
check("returned pair achieves that distance", approx(_dist(pp, qq), dmin))

# --- a grid: closest pair is the unit spacing ------------------------------
grid = [(x, y) for x in range(10) for y in range(10)]
_, _, dg = closest_pair(grid)
check("unit grid closest pair is 1", approx(dg, 1.0))

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all closest_pair tests passed")

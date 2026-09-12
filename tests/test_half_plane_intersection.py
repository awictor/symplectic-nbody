"""Tests for half_plane_intersection: feasible-region polygon vs brute grid sampling."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from half_plane_intersection import (intersect_half_planes, is_feasible, polygon_area,
                                     contains_point, clip_polygon, _inside,
                                     brute_feasible_point)

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


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


# --- known regions ----------------------------------------------------------
# unit box 0<=x<=1, 0<=y<=1
box = [(-1, 0, 0), (1, 0, 1), (0, -1, 0), (0, 1, 1)]
poly = intersect_half_planes(box)
check("unit box has area 1", close(polygon_area(poly), 1.0))
check("unit box has 4 vertices", len(poly) == 4)

# triangle x>=0, y>=0, x+y<=1
tri = [(-1, 0, 0), (0, -1, 0), (1, 1, 1)]
check("unit triangle has area 1/2", close(polygon_area(intersect_half_planes(tri)), 0.5))

# a 2x3 rectangle
rect = [(-1, 0, 0), (1, 0, 2), (0, -1, 0), (0, 1, 3)]
check("2x3 rectangle has area 6", close(polygon_area(intersect_half_planes(rect)), 6.0))

# infeasible: x <= 0 and x >= 1
check("x<=0 and x>=1 is infeasible", not is_feasible([(1, 0, 0), (-1, 0, -1)]))
# feasible: overlapping constraints
check("overlapping constraints are feasible", is_feasible([(1, 0, 5), (-1, 0, 0), (0, 1, 5), (0, -1, 0)]))

# --- feasibility matches brute grid ----------------------------------------
rng = LCG(2026)
feas_ok = True
for _ in range(400):
    hps = []
    for _ in range(rng.randint(2, 6)):
        a = rng.randint(-3, 3)
        b = rng.randint(-3, 3)
        if a == 0 and b == 0:
            a = 1
        c = rng.randint(-5, 5)
        hps.append((a, b, c))
    computed = is_feasible(hps, bound=100)
    brute = brute_feasible_point(hps, lo=-8, hi=8, steps=120) is not None
    # the grid can miss a tiny feasible sliver, so only require: if brute finds one, we must agree
    if brute and not computed:
        feas_ok = False
        print(f"  feasibility mismatch (brute yes, computed no): {hps}")
        break
check("if a brute grid finds a feasible point, the polygon is nonempty (400 systems)", feas_ok)

# --- every polygon vertex satisfies all constraints ------------------------
rng = LCG(4242)
vertex_ok = True
for _ in range(400):
    hps = []
    for _ in range(rng.randint(3, 6)):
        a = rng.randint(-3, 3)
        b = rng.randint(-3, 3)
        if a == 0 and b == 0:
            b = 1
        c = rng.randint(0, 8)      # c>=0 keeps origin feasible -> bounded-ish regions
        hps.append((a, b, c))
    poly = intersect_half_planes(hps, bound=50)
    for v in poly:
        if not all(_inside(v, hp) for hp in hps):
            vertex_ok = False
            break
    if not vertex_ok:
        break
check("every feasible-polygon vertex satisfies all constraints (400 systems)", vertex_ok)

# --- polygon membership matches the inequalities on a grid -----------------
# use bounded regions (a box constraint always present) and check grid points
rng = LCG(777)
member_ok = True
for _ in range(120):
    hps = [(-1, 0, 5), (1, 0, 5), (0, -1, 5), (0, 1, 5)]   # box [-5,5]^2
    for _ in range(rng.randint(1, 4)):
        a = rng.randint(-2, 2)
        b = rng.randint(-2, 2)
        if a == 0 and b == 0:
            a = 1
        c = rng.randint(-3, 3)
        hps.append((a, b, c))
    poly = intersect_half_planes(hps, bound=100)
    # sample grid points; feasible-by-inequalities must match inside-polygon
    steps = 25
    for i in range(steps + 1):
        x = -5 + 10 * i / steps
        for j in range(steps + 1):
            y = -5 + 10 * j / steps
            feasible = all(_inside((x, y), hp) for hp in hps)
            inside = contains_point(poly, (x, y), tol=1e-3) if poly else False
            # allow disagreement only within a thin boundary band
            if feasible != inside:
                # check it's near a boundary (some constraint nearly tight)
                near_boundary = any(abs(a * x + b * y - c) < 0.3 for (a, b, c) in hps)
                if not near_boundary:
                    member_ok = False
                    break
        if not member_ok:
            break
    if not member_ok:
        break
check("polygon membership matches the inequalities away from boundaries (120 systems)", member_ok)

# --- clipping a polygon by a trivially-true half-plane is a no-op ----------
square = [(0, 0), (2, 0), (2, 2), (0, 2)]
check("clipping by a satisfied half-plane keeps the polygon",
      clip_polygon(square, (1, 0, 100)) == square)
check("clipping by a violated half-plane empties the polygon",
      clip_polygon(square, (1, 0, -100)) == [])

# --- adding a constraint never grows the area ------------------------------
rng = LCG(555)
monotone_ok = True
for _ in range(150):
    hps = [(-1, 0, 5), (1, 0, 5), (0, -1, 5), (0, 1, 5)]
    a0 = polygon_area(intersect_half_planes(hps, bound=100))
    hps2 = hps + [(rng.randint(-2, 2) or 1, rng.randint(-2, 2), rng.randint(-2, 2))]
    a1 = polygon_area(intersect_half_planes(hps2, bound=100))
    if a1 > a0 + 1e-6:
        monotone_ok = False
        break
check("adding a constraint never increases the feasible area", monotone_ok)

# --- a half-plane region reduces to the expected line/point limits ---------
# intersection giving a single point: x<=0, x>=0, y<=0, y>=0 -> the origin (area 0)
pt = intersect_half_planes([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)])
check("four constraints pinning the origin give ~zero area", polygon_area(pt) < 1e-6)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all half_plane_intersection tests passed")

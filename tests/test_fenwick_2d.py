"""Tests for fenwick_2d: point updates + rectangle sums vs a brute-force grid."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fenwick_2d import Fenwick2D, from_matrix, BruteGrid

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


# --- known small case -------------------------------------------------------
fw = Fenwick2D(2, 3)
for r, c, d in [(0, 0, 1), (0, 1, 2), (0, 2, 3), (1, 0, 4), (1, 1, 5), (1, 2, 6)]:
    fw.update(r, c, d)
check("full-grid sum is 21", fw.rectangle_sum(0, 0, 1, 2) == 21)
check("first row sum is 6", fw.rectangle_sum(0, 0, 0, 2) == 6)
check("last column sum is 9", fw.rectangle_sum(0, 2, 1, 2) == 9)
check("single cell (1,1) is 5", fw.point_value(1, 1) == 5)
check("empty rectangle (r1>r2) is 0", fw.rectangle_sum(1, 0, 0, 2) == 0)

# --- from_matrix ------------------------------------------------------------
m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
fm = from_matrix(m)
check("from_matrix full sum is 45", fm.rectangle_sum(0, 0, 2, 2) == 45)
check("from_matrix middle cell is 5", fm.point_value(1, 1) == 5)
check("from_matrix center block (1,1)-(2,2) is 28", fm.rectangle_sum(1, 1, 2, 2) == 28)

# --- random updates then all rectangle queries vs brute --------------------
rng = LCG(2026)
rect_ok = True
for _ in range(200):
    R = rng.randint(1, 8)
    C = rng.randint(1, 8)
    fw = Fenwick2D(R, C)
    bg = BruteGrid(R, C)
    # random updates
    for _ in range(rng.randint(1, 20)):
        r = rng.randint(0, R - 1)
        c = rng.randint(0, C - 1)
        d = rng.randint(-9, 9)
        fw.update(r, c, d)
        bg.update(r, c, d)
    # all rectangles
    for r1 in range(R):
        for c1 in range(C):
            for r2 in range(r1, R):
                for c2 in range(c1, C):
                    if fw.rectangle_sum(r1, c1, r2, c2) != bg.rectangle_sum(r1, c1, r2, c2):
                        rect_ok = False
                        break
                if not rect_ok:
                    break
            if not rect_ok:
                break
        if not rect_ok:
            break
    if not rect_ok:
        print(f"  rectangle mismatch on R={R} C={C}")
        break
check("all rectangle sums match brute after random updates (200 grids)", rect_ok)

# --- interleaved updates and queries ---------------------------------------
rng = LCG(4242)
interleave_ok = True
for _ in range(200):
    R = rng.randint(1, 10)
    C = rng.randint(1, 10)
    fw = Fenwick2D(R, C)
    bg = BruteGrid(R, C)
    for _ in range(40):
        if rng.rand() % 2 == 0:
            r = rng.randint(0, R - 1)
            c = rng.randint(0, C - 1)
            d = rng.randint(-5, 5)
            fw.update(r, c, d)
            bg.update(r, c, d)
        else:
            r1 = rng.randint(0, R - 1)
            c1 = rng.randint(0, C - 1)
            r2 = rng.randint(r1, R - 1)
            c2 = rng.randint(c1, C - 1)
            if fw.rectangle_sum(r1, c1, r2, c2) != bg.rectangle_sum(r1, c1, r2, c2):
                interleave_ok = False
                break
    if not interleave_ok:
        break
check("interleaved updates and queries stay correct (200 grids)", interleave_ok)

# --- prefix sums match ------------------------------------------------------
rng = LCG(777)
prefix_ok = True
for _ in range(150):
    R = rng.randint(1, 9)
    C = rng.randint(1, 9)
    fw = Fenwick2D(R, C)
    bg = BruteGrid(R, C)
    for _ in range(rng.randint(1, 25)):
        r = rng.randint(0, R - 1)
        c = rng.randint(0, C - 1)
        d = rng.randint(0, 10)
        fw.update(r, c, d)
        bg.update(r, c, d)
    for r in range(-1, R):
        for c in range(-1, C):
            if fw.prefix_sum(r, c) != bg.prefix_sum(r, c):
                prefix_ok = False
                break
        if not prefix_ok:
            break
    if not prefix_ok:
        break
check("prefix sums match brute (including negative/empty prefixes)", prefix_ok)

# --- point-value reads match after updates ---------------------------------
rng = LCG(555)
point_ok = True
for _ in range(150):
    R = rng.randint(1, 8)
    C = rng.randint(1, 8)
    fw = Fenwick2D(R, C)
    bg = BruteGrid(R, C)
    for _ in range(30):
        r = rng.randint(0, R - 1)
        c = rng.randint(0, C - 1)
        d = rng.randint(-7, 7)
        fw.update(r, c, d)
        bg.update(r, c, d)
    for r in range(R):
        for c in range(C):
            if fw.point_value(r, c) != bg.point_value(r, c):
                point_ok = False
                break
        if not point_ok:
            break
    if not point_ok:
        break
check("point-value reads match brute after updates", point_ok)

# --- larger grid: a big instance solves and matches spot checks ------------
rng = LCG(31337)
R = C = 200
fw = Fenwick2D(R, C)
bg = BruteGrid(R, C)
for _ in range(2000):
    r = rng.randint(0, R - 1)
    c = rng.randint(0, C - 1)
    d = rng.randint(-5, 5)
    fw.update(r, c, d)
    bg.update(r, c, d)
spot_ok = True
for _ in range(50):
    r1 = rng.randint(0, R - 1)
    c1 = rng.randint(0, C - 1)
    r2 = rng.randint(r1, R - 1)
    c2 = rng.randint(c1, C - 1)
    if fw.rectangle_sum(r1, c1, r2, c2) != bg.rectangle_sum(r1, c1, r2, c2):
        spot_ok = False
        break
check("200x200 grid, 2000 updates: rectangle spot-checks match brute", spot_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all fenwick_2d tests passed")

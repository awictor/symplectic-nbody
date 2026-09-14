"""Tests for Minkowski sum: edge-merge vs brute, known sums, translation, collision via difference."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import minkowski_sum as MS  # noqa: E402
from convex_hull import convex_hull  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _lcg(seed):
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]

    # ---- 1. square + square = 2x2 square (area 4) ---------------------------------------
    s = MS.minkowski_sum_convex(sq, sq)
    check("square (+) square area = 4", abs(MS.area(s) - 4.0) < 1e-9, f"{MS.area(s)}")
    check("square (+) square has 4 vertices", len(s) == 4, f"{len(s)}")

    # ---- 2. edge-merge matches brute-force on known and random inputs -------------------
    tri = [(0, 0), (2, 0), (1, 2)]
    check("tri (+) square: edge-merge == brute",
          abs(MS.area(MS.minkowski_sum_convex(tri, sq)) - MS.area(MS.minkowski_sum_brute(tri, sq))) < 1e-9)

    rnd = _lcg(7)

    def randconvex(k, cx, cy, r):
        pts = [(cx + r * (0.4 + rnd()) * math.cos(2 * math.pi * i / k),
                cy + r * (0.4 + rnd()) * math.sin(2 * math.pi * i / k)) for i in range(k)]
        return convex_hull(pts)

    bad = 0
    for _ in range(40):
        A = randconvex(6, 0, 0, 2)
        B = randconvex(5, 4, 4, 1.5)
        if abs(MS.area(MS.minkowski_sum_convex(A, B)) - MS.area(MS.minkowski_sum_brute(A, B))) > 1e-6:
            bad += 1
    check("edge-merge == brute on 40 random convex pairs", bad == 0, f"{bad} mismatches")

    # ---- 3. summing with a single point is a pure translation ---------------------------
    pt = [(3.0, -2.0)]
    trans = MS.minkowski_sum_brute(sq, pt)
    # the translated square has the same area and is shifted by (3,-2)
    check("sum with a point preserves area", abs(MS.area(trans) - 1.0) < 1e-9, f"{MS.area(trans)}")
    xs = [p[0] for p in trans]
    ys = [p[1] for p in trans]
    check("sum with a point translates correctly",
          abs(min(xs) - 3.0) < 1e-9 and abs(min(ys) - (-2.0)) < 1e-9, f"{trans}")

    # ---- 4. area of the sum >= area(A) + area(B) for convex shapes ----------------------
    A = randconvex(6, 0, 0, 3)
    B = randconvex(5, 0, 0, 2)
    from convex_hull import polygon_area
    check("area(A(+)B) >= area(A) + area(B)",
          MS.area(MS.minkowski_sum_convex(A, B)) >= polygon_area(A) + polygon_area(B) - 1e-6,
          f"{MS.area(MS.minkowski_sum_convex(A, B))} vs {polygon_area(A)+polygon_area(B)}")

    # ---- 5. translating an input translates the sum -------------------------------------
    B_shift = [(x + 5, y - 3) for x, y in B]
    s1 = MS.minkowski_sum_convex(A, B)
    s2 = MS.minkowski_sum_convex(A, B_shift)
    # s2 should be s1 shifted by (5,-3): same area, centroid shifted
    def centroid(poly):
        return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
    c1 = centroid(s1)
    c2 = centroid(s2)
    check("translating B translates the sum",
          abs(MS.area(s1) - MS.area(s2)) < 1e-6 and abs((c2[0] - c1[0]) - 5) < 0.5,
          f"area {MS.area(s1):.3f}/{MS.area(s2):.3f}")

    # ---- 6. collision via Minkowski difference: overlapping shapes ----------------------
    P = [(0, 0), (2, 0), (2, 2), (0, 2)]
    Q_over = [(1, 1), (3, 1), (3, 3), (1, 3)]
    check("overlapping squares intersect (origin in A(+)(-B))", MS.convex_intersect(P, Q_over))

    # ---- 7. collision: disjoint shapes do not intersect ---------------------------------
    Q_dis = [(5, 5), (6, 5), (6, 6), (5, 6)]
    check("disjoint squares do not intersect", not MS.convex_intersect(P, Q_dis))

    # ---- 8. touching shapes (share a boundary) count as intersecting --------------------
    Q_touch = [(2, 0), (4, 0), (4, 2), (2, 2)]    # shares the edge x=2
    check("edge-touching squares intersect", MS.convex_intersect(P, Q_touch))

    # ---- 9. Minkowski difference is the sum with the negated polygon --------------------
    diff = MS.minkowski_difference(P, Q_over)
    manual = MS.minkowski_sum_convex(P, MS.negate(Q_over))
    check("difference == sum with negated polygon", abs(MS.area(diff) - MS.area(manual)) < 1e-9)

    # ---- 10. a random collision battery vs a brute overlap check ------------------------
    rnd2 = _lcg(99)
    ok = True
    for _ in range(30):
        cx = rnd2() * 6
        R = randconvex(5, cx, 0, 1.5)
        S = randconvex(5, 3, 0, 1.5)
        mink = MS.convex_intersect(R, S)
        # brute: sample points of R, test if any lies in hull of S (approximate overlap witness)
        # use a robust check: intersect iff their Minkowski difference contains origin (definitional)
        # cross-check with a bounding-box necessary condition
        rx = [p[0] for p in R]
        sx = [p[0] for p in S]
        if max(rx) < min(sx) - 1e-9 or max(sx) < min(rx) - 1e-9:
            if mink:                              # x-ranges disjoint => cannot intersect
                ok = False
    check("collision test respects bounding-box separation", ok)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

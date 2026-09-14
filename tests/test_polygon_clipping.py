"""Tests for Sutherland-Hodgman polygon clipping: containment, area bounds, idempotence, known intersections."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import polygon_clipping as PC  # noqa: E402


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
    window = [(2, 2), (8, 2), (8, 8), (2, 8)]     # square [2,8]^2, area 36

    # ---- 1. a big square clipped to the window becomes the window -----------------------
    big = [(0, 0), (10, 0), (10, 10), (0, 10)]
    c = PC.clip(big, window)
    check("big square clipped to window has window's area", abs(PC.area(c) - 36.0) < 1e-9,
          f"{PC.area(c)}")

    # ---- 2. a polygon entirely inside is returned unchanged (same area) -----------------
    small = [(3, 3), (5, 3), (5, 5), (3, 5)]
    cs = PC.clip(small, window)
    check("inside polygon area unchanged", abs(PC.area(cs) - 4.0) < 1e-9, f"{PC.area(cs)}")

    # ---- 3. a polygon entirely outside is clipped to nothing ----------------------------
    outside = [(20, 20), (22, 20), (22, 22), (20, 22)]
    check("outside polygon clips to empty", PC.clip(outside, window) == [])

    # ---- 4. clipping is idempotent: clipping the window by itself is the window ---------
    cw = PC.clip(window, window)
    check("window clipped by itself is idempotent", abs(PC.area(cw) - 36.0) < 1e-9, f"{PC.area(cw)}")

    # ---- 5. triangle clipped by a rectangle: hand-computed area -------------------------
    tri = [(0, 0), (10, 0), (5, 10)]              # area 50
    cr = PC.clip_rectangle(tri, 0, 0, 10, 5)      # keep y in [0,5]: trapezoid, widths 10 and 5
    check("triangle clipped to y<=5 trapezoid area 37.5", abs(PC.area(cr) - 37.5) < 1e-9,
          f"{PC.area(cr)}")

    # ---- 6. clipped area never exceeds subject area or window area ----------------------
    import math
    rnd = _lcg(7)
    ok = True
    for _ in range(50):
        # random CONVEX subject: points sorted by angle around their centroid (simple polygon)
        pts = [(rnd() * 12 - 1, rnd() * 12 - 1) for _ in range(5)]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        subj = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        clipped = PC.clip(subj, window)
        a_clip = PC.area(clipped)
        if a_clip > PC.area(subj) + 1e-6 or a_clip > 36.0 + 1e-6:
            ok = False
            break
    check("clipped area <= subject area and <= window area", ok)

    # ---- 7. every clipped vertex lies inside the (closed) window ------------------------
    subj = [(-1, 5), (5, -1), (11, 5), (5, 11)]   # diamond overlapping the window
    clipped = PC.clip(subj, window)
    check("all clipped vertices inside the window",
          all(PC.point_in_convex(p, window) for p in clipped), f"{clipped}")

    # ---- 8. clipping against a triangular (convex) window -------------------------------
    tri_win = [(0, 0), (10, 0), (0, 10)]          # right triangle, area 50
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    ct = PC.clip(sq, tri_win)
    check("square clipped by triangle window == triangle", abs(PC.area(ct) - 50.0) < 1e-9,
          f"{PC.area(ct)}")

    # ---- 9. rectangle-rectangle overlap area --------------------------------------------
    r1 = [(0, 0), (6, 0), (6, 6), (0, 6)]
    overlap = PC.clip_rectangle(r1, 3, 3, 10, 10)  # overlap is [3,6]^2, area 9
    check("rectangle-rectangle overlap area 9", abs(PC.area(overlap) - 9.0) < 1e-9,
          f"{PC.area(overlap)}")

    # ---- 10. point-in-convex sanity -----------------------------------------------------
    check("center is inside window", PC.point_in_convex((5, 5), window))
    check("far point is outside window", not PC.point_in_convex((100, 100), window))

    # ---- 11. half-overlap: square straddling one window edge ----------------------------
    straddle = [(5, 5), (12, 5), (12, 7), (5, 7)]  # extends past x=8; kept part x in [5,8]
    cstr = PC.clip(straddle, window)
    check("straddling square keeps only in-window part (area 6)", abs(PC.area(cstr) - 6.0) < 1e-9,
          f"{PC.area(cstr)}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

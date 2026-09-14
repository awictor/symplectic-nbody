"""Tests for Catmull-Rom splines: interpolation, C1 continuity, Hermite reduction, centripetal no-overshoot."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import catmull_rom as CR  # noqa: E402


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


def main():
    pts = [[0, 0], [1, 2], [3, 3], [4, 0], [6, 1], [7, 3]]

    # ---- 1. interpolation: the curve passes through every control point -----------------
    for kind in ("uniform", "centripetal", "chordal"):
        c = CR.CatmullRom(pts, kind)
        err = max(max(abs(c.evaluate(i)[d] - pts[i][d]) for d in range(2)) for i in range(len(pts)))
        check(f"{kind} interpolates all control points", err < 1e-9, f"err {err}")

    # ---- 2. segment endpoints match the control points ----------------------------------
    c = CR.centripetal(pts)
    ok = True
    for seg in range(c.n_segments()):
        a = c.evaluate_segment(seg, 0.0)
        b = c.evaluate_segment(seg, 1.0)
        if max(abs(a[d] - pts[seg][d]) for d in range(2)) > 1e-9:
            ok = False
        if max(abs(b[d] - pts[seg + 1][d]) for d in range(2)) > 1e-9:
            ok = False
    check("segment endpoints equal the control points", ok)

    # ---- 3. C1 continuity: tangent direction matches across segment joins ---------------
    ok = True
    for seg in range(c.n_segments() - 1):
        ta = c.tangent_segment(seg, 1.0)
        tb = c.tangent_segment(seg + 1, 0.0)
        na = math.hypot(*ta)
        nb = math.hypot(*tb)
        if na < 1e-9 or nb < 1e-9:
            continue
        cos = (ta[0] * tb[0] + ta[1] * tb[1]) / (na * nb)
        if cos < 1 - 1e-4:
            ok = False
    check("C1: tangent directions continuous across joins", ok)

    # ---- 4. position continuity across joins (C0) ---------------------------------------
    ok = True
    for seg in range(c.n_segments() - 1):
        end = c.evaluate_segment(seg, 1.0)
        start = c.evaluate_segment(seg + 1, 0.0)
        if max(abs(end[d] - start[d]) for d in range(2)) > 1e-9:
            ok = False
    check("C0: position continuous across joins", ok)

    # ---- 5. uniform reduces to the classic Hermite tangent = (P_{i+1}-P_{i-1})/2 --------
    cu = CR.uniform(pts)
    ok = True
    for i in range(1, len(pts) - 1):
        # classic tangent at interior point i
        expected = [(pts[i + 1][d] - pts[i - 1][d]) / 2 for d in range(2)]
        # numerical tangent from the segment starting at i, scaled by segment param length (=1 uniform)
        tang = cu.tangent_segment(i, 0.0)
        if max(abs(tang[d] - expected[d]) for d in range(2)) > 1e-3:
            ok = False
    check("uniform tangents == classic (P_{i+1}-P_{i-1})/2", ok)

    # ---- 6. collinear points stay on the line -------------------------------------------
    line = [[0, 0], [1, 1], [2, 2], [4, 4], [5, 5]]
    for kind in ("uniform", "centripetal", "chordal"):
        cl = CR.CatmullRom(line, kind)
        off = 0.0
        for p in cl.sample(30):
            off = max(off, abs(p[1] - p[0]))
        check(f"{kind}: collinear points stay straight", off < 1e-9, f"off {off}")

    # ---- 7. centripetal does not overshoot a segment where uniform does -----------------
    # Yuksel-Schaefer-Keyser style: tight then wide spacing near a corner.
    sharp = [[0, 0], [10, 0], [11, 5], [12, 0], [22, 0]]

    def seg_x_overshoot(kind):
        cc = CR.CatmullRom(sharp, kind)
        xs = [cc.evaluate_segment(1, k / 60)[0] for k in range(61)]
        return max(0.0, 10 - min(xs)) + max(0.0, max(xs) - 11)

    uni_over = seg_x_overshoot("uniform")
    cen_over = seg_x_overshoot("centripetal")
    check("uniform overshoots the segment box", uni_over > 1e-3, f"{uni_over}")
    check("centripetal does not overshoot", cen_over < 1e-9, f"{cen_over}")

    # ---- 8. reversing the points reverses the curve -------------------------------------
    c = CR.centripetal(pts)
    cr = CR.centripetal(list(reversed(pts)))
    n = c.n_segments()
    ok = True
    for k in range(51):
        u = k / 50 * n
        a = c.evaluate(u)
        b = cr.evaluate(n - u)
        if max(abs(a[d] - b[d]) for d in range(2)) > 1e-7:
            ok = False
    check("reversing control points reverses the curve", ok)

    # ---- 9. 3D control points -----------------------------------------------------------
    c3 = CR.centripetal([[0, 0, 0], [1, 2, 1], [3, 1, 4], [4, 0, 2], [5, 3, 1]])
    p = c3.evaluate(1.5)
    check("3D evaluation returns 3D point", len(p) == 3)
    check("3D interpolates control points",
          all(abs(c3.evaluate(2)[d] - [3, 1, 4][d]) < 1e-9 for d in range(3)))

    # ---- 10. sample returns a connected polyline of the right span ----------------------
    s = c.sample(10)
    check("sample starts at first point",
          all(abs(s[0][d] - pts[0][d]) < 1e-9 for d in range(2)))
    check("sample ends at last point",
          all(abs(s[-1][d] - pts[-1][d]) < 1e-9 for d in range(2)))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

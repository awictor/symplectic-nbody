"""Tests for NURBS: exact circle/ellipse, rational partition of unity, B-spline reduction, endpoint clamp."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import nurbs as NU  # noqa: E402
import b_spline  # noqa: E402


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
    # ---- 1. THE marquee test: the NURBS circle is exactly a circle ----------------------
    ctrl, w, p, kn = NU.circle((0.0, 0.0), 1.0)
    max_r_err = 0.0
    for k in range(500):
        u = k / 499
        pt = NU.evaluate(ctrl, w, p, u, kn)
        max_r_err = max(max_r_err, abs(math.hypot(pt[0], pt[1]) - 1.0))
    check("NURBS unit circle lies exactly on r=1", max_r_err < 1e-12, f"max err {max_r_err}")

    # ---- 2. circle at an offset center with a different radius ---------------------------
    ctrl2, w2, p2, kn2 = NU.circle((3.0, -2.0), 2.5)
    err2 = 0.0
    for k in range(300):
        u = k / 299
        pt = NU.evaluate(ctrl2, w2, p2, u, kn2)
        err2 = max(err2, abs(math.hypot(pt[0] - 3.0, pt[1] + 2.0) - 2.5))
    check("offset circle (center (3,-2), r=2.5) exact", err2 < 1e-12, f"{err2}")

    # ---- 3. the circle actually goes all the way around (covers all quadrants) ----------
    pts = NU.sample(ctrl, w, p, kn, 200)
    angles = [math.atan2(pt[1], pt[0]) for pt in pts]
    quad = set()
    for pt in pts:
        quad.add((pt[0] >= 0, pt[1] >= 0))
    check("circle covers all four quadrants", len(quad) == 4)
    check("circle closes (start ~ end)",
          math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-9)

    # ---- 4. rational basis is a partition of unity and non-negative ---------------------
    pu_err = 0.0
    min_b = 1e9
    for k in range(300):
        u = k / 299
        R = NU.rational_basis(w, p, u, kn, len(ctrl))
        pu_err = max(pu_err, abs(sum(R) - 1.0))
        min_b = min(min_b, min(R))
    check("rational basis partition of unity", pu_err < 1e-12, f"{pu_err}")
    check("rational basis non-negative", min_b >= -1e-15, f"min {min_b}")

    # ---- 5. unit weights => NURBS reduces to an ordinary B-spline -----------------------
    cc = [[0, 0], [1, 3], [3, 3], [4, 0], [6, 2], [7, 4]]
    deg = 3
    knb = b_spline.clamped_knots(len(cc), deg)
    wones = [1.0] * len(cc)
    md = 0.0
    for k in range(200):
        u = k / 199
        a = NU.evaluate(cc, wones, deg, u, knb)
        b = b_spline.evaluate(cc, deg, u, knb)
        md = max(md, max(abs(a[c] - b[c]) for c in range(2)))
    check("unit-weight NURBS == B-spline", md < 1e-10, f"maxdiff {md}")

    # ---- 6. clamped NURBS interpolates its endpoints ------------------------------------
    ww = [1.0, 2.0, 0.5, 3.0, 1.0, 1.5]
    start = NU.evaluate(cc, ww, deg, knb[deg], knb)
    end = NU.evaluate(cc, ww, deg, knb[len(cc)], knb)
    check("clamped NURBS starts at first control point (any weights)",
          all(abs(start[c] - cc[0][c]) < 1e-9 for c in range(2)), f"{start}")
    check("clamped NURBS ends at last control point (any weights)",
          all(abs(end[c] - cc[-1][c]) < 1e-9 for c in range(2)), f"{end}")

    # ---- 7. increasing a control point's weight pulls the curve toward it ----------------
    mid_u = (knb[deg] + knb[len(cc)]) / 2
    base = NU.evaluate(cc, [1.0] * len(cc), deg, mid_u, knb)
    heavy = list([1.0] * len(cc))
    heavy[2] = 8.0                                   # heavily weight control point 2
    pulled = NU.evaluate(cc, heavy, deg, mid_u, knb)
    d_base = math.dist(base, cc[2])
    d_pulled = math.dist(pulled, cc[2])
    check("raising a weight pulls the curve toward that point", d_pulled < d_base,
          f"base {d_base:.3f} pulled {d_pulled:.3f}")

    # ---- 8. ellipse satisfies its implicit equation x^2/a^2 + y^2/b^2 = 1 ---------------
    ce, we, pe, kne = NU.ellipse((0.0, 0.0), 3.0, 1.5)
    eerr = 0.0
    for k in range(300):
        u = k / 299
        pt = NU.evaluate(ce, we, pe, u, kne)
        eerr = max(eerr, abs(pt[0] ** 2 / 9.0 + pt[1] ** 2 / 2.25 - 1.0))
    check("NURBS ellipse satisfies implicit equation", eerr < 1e-12, f"{eerr}")

    # ---- 9. circle arc length == 2*pi*r (numerically) -----------------------------------
    circ = NU.sample(ctrl, w, p, kn, 2000)
    L = sum(math.dist(circ[i], circ[i + 1]) for i in range(len(circ) - 1))
    check("circle arc length ~ 2*pi", abs(L - 2 * math.pi) < 1e-4, f"L {L}")

    # ---- 10. 3D NURBS control points work (circle lifted to z=const plane) --------------
    c3 = [[p3[0], p3[1], 1.0] for p3 in ctrl]
    pt3 = NU.evaluate(c3, w, p, 0.3, kn)
    check("3D NURBS returns 3D point on the lifted circle",
          len(pt3) == 3 and abs(math.hypot(pt3[0], pt3[1]) - 1.0) < 1e-12 and abs(pt3[2] - 1.0) < 1e-12)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

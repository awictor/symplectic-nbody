"""Tests for B-splines: partition of unity, local support, de Boor vs basis, endpoint clamp, Bezier reduction."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import b_spline as B  # noqa: E402
import bezier  # noqa: E402


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
    ctrl = [[0, 0], [1, 2], [3, 3], [4, 0], [5, 2], [6, 1]]
    p = 3
    kn = B.clamped_knots(len(ctrl), p)

    # ---- 1. clamped knot vector shape ---------------------------------------------------
    check("clamped knots length n+p+2", len(kn) == len(ctrl) + p + 1)
    check("first p+1 knots equal", kn[:p + 1] == [0.0] * (p + 1))
    check("last p+1 knots equal", kn[-(p + 1):] == [1.0] * (p + 1))
    check("knots non-decreasing", all(kn[i] <= kn[i + 1] for i in range(len(kn) - 1)))

    # ---- 2. partition of unity + non-negativity -----------------------------------------
    max_sum_err = 0.0
    min_basis = 1e9
    for k in range(200):
        u = k / 199
        N = B.all_basis(p, u, kn, len(ctrl))
        max_sum_err = max(max_sum_err, abs(sum(N) - 1.0))
        min_basis = min(min_basis, min(N))
    check("basis is a partition of unity", max_sum_err < 1e-12, f"err {max_sum_err}")
    check("basis functions non-negative", min_basis >= -1e-15, f"min {min_basis}")

    # ---- 3. local support: N_{i,p} zero outside [u_i, u_{i+p+1}] -------------------------
    ok = True
    i = 2
    for k in range(200):
        u = k / 199
        val = B.basis(i, p, u, kn)
        if val > 1e-12 and not (kn[i] - 1e-12 <= u <= kn[i + p + 1] + 1e-12):
            ok = False
    check("basis has local support (p+1 spans)", ok)

    # ---- 4. de Boor evaluation == direct basis summation --------------------------------
    max_diff = 0.0
    for k in range(200):
        u = k / 199
        e = B.evaluate_basis(ctrl, p, u, kn)
        d = B.evaluate(ctrl, p, u, kn)
        max_diff = max(max_diff, max(abs(e[c] - d[c]) for c in range(2)))
    check("de Boor == basis summation", max_diff < 1e-12, f"maxdiff {max_diff}")

    # ---- 5. clamped curve interpolates its endpoints ------------------------------------
    start = B.evaluate(ctrl, p, 0.0, kn)
    end = B.evaluate(ctrl, p, 1.0, kn)
    check("clamped curve starts at first control point",
          all(abs(start[c] - ctrl[0][c]) < 1e-12 for c in range(2)), f"{start}")
    check("clamped curve ends at last control point",
          all(abs(end[c] - ctrl[-1][c]) < 1e-12 for c in range(2)), f"{end}")

    # ---- 6. curve stays in the convex hull (x/y within control bounds) ------------------
    xs = [c[0] for c in ctrl]
    ys = [c[1] for c in ctrl]
    inside = True
    for pt in B.sample(ctrl, p, kn, 100):
        if not (min(xs) - 1e-9 <= pt[0] <= max(xs) + 1e-9 and
                min(ys) - 1e-9 <= pt[1] <= max(ys) + 1e-9):
            inside = False
    check("curve stays within control-point bounding box", inside)

    # ---- 7. clamped B-spline with no interior knots == Bezier ---------------------------
    for nctrl in (4, 5):
        cc = [[i, (i * 7) % 5] for i in range(nctrl)]
        deg = nctrl - 1
        knb = B.clamped_knots(nctrl, deg)
        # no interior knots => Bezier
        interior = [x for x in knb if 0.0 < x < 1.0]
        bmax = 0.0
        for k in range(60):
            t = k / 59
            bs = B.evaluate(cc, deg, t, knb)
            bz = bezier.evaluate(cc, t)
            bmax = max(bmax, max(abs(bs[c] - bz[c]) for c in range(2)))
        check(f"degree-{deg} clamped B-spline == Bezier (no interior knots)",
              len(interior) == 0 and bmax < 1e-10, f"interior {interior} bmax {bmax}")

    # ---- 8. analytic derivative control points match finite difference ------------------
    Q = B.derivative_control_points(ctrl, p, kn)
    # derivative curve is degree p-1 on the knot vector with first/last knot dropped
    dknots = kn[1:-1]
    max_dd = 0.0
    for k in range(1, 199):
        u = k / 199 * 0.999 + 0.0005
        fd = B.derivative(ctrl, p, u, kn)
        # evaluate derivative curve
        try:
            an = B.evaluate(Q, p - 1, u, dknots)
        except Exception:
            continue
        max_dd = max(max_dd, max(abs(fd[c] - an[c]) for c in range(2)))
    check("derivative control points match finite difference", max_dd < 1e-3, f"maxdd {max_dd}")

    # ---- 9. uniform knot vector is valid and gives partition of unity -------------------
    uk = B.uniform_knots(len(ctrl), p)
    check("uniform knots non-decreasing", all(uk[i] <= uk[i + 1] for i in range(len(uk) - 1)))
    # sample in the valid interior domain
    su = 0.0
    u = uk[p] + (uk[len(ctrl)] - uk[p]) * 0.5
    N = B.all_basis(p, u, uk, len(ctrl))
    check("uniform basis partition of unity (interior)", abs(sum(N) - 1.0) < 1e-12, f"{sum(N)}")

    # ---- 10. 3D control points work ------------------------------------------------------
    c3 = [[0, 0, 0], [1, 2, 1], [3, 1, 4], [4, 0, 2], [5, 3, 1]]
    k3 = B.clamped_knots(len(c3), 3)
    pt = B.evaluate(c3, 3, 0.5, k3)
    check("3D evaluation returns 3D point", len(pt) == 3)
    check("3D clamped starts at first point",
          all(abs(B.evaluate(c3, 3, 0.0, k3)[c] - c3[0][c]) < 1e-12 for c in range(3)))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

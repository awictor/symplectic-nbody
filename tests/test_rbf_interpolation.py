"""Tests for RBF interpolation: node reproduction, smooth recovery, thin-plate linear reproduction, refinement."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rbf_interpolation as R  # noqa: E402


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


KINDS = ["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate", "linear", "cubic"]


def main():
    rnd = _lcg(7)
    pts = [[rnd() * 2 - 1, rnd() * 2 - 1] for _ in range(40)]
    f = lambda p: math.exp(-(p[0] ** 2 + p[1] ** 2))
    vals = [f(p) for p in pts]

    # ---- 1. every kernel reproduces the data values at the nodes ------------------------
    for kind in KINDS:
        rbf = R.RBFInterpolator(pts, vals, kind, epsilon=1.5)
        nerr = max(abs(rbf.evaluate(pts[i]) - vals[i]) for i in range(len(pts)))
        check(f"{kind} reproduces node values", nerr < 1e-8, f"node err {nerr:.2e}")

    # ---- 2. smooth interior recovery of the Gaussian bump -------------------------------
    for kind in ("gaussian", "multiquadric"):
        rbf = R.RBFInterpolator(pts, vals, kind, epsilon=1.5)
        ie = 0.0
        for _ in range(300):
            q = [rnd() * 1.2 - 0.6, rnd() * 1.2 - 0.6]
            ie = max(ie, abs(rbf.evaluate(q) - f(q)))
        check(f"{kind} recovers the bump in the interior", ie < 5e-3, f"interior err {ie:.2e}")

    # ---- 3. thin-plate spline reproduces any linear function EXACTLY --------------------
    lin = lambda p: 3.0 + 2.0 * p[0] - 1.5 * p[1]
    vlin = [lin(p) for p in pts]
    rbf = R.RBFInterpolator(pts, vlin, "thin_plate")
    lerr = 0.0
    for _ in range(300):
        q = [rnd() * 4 - 2, rnd() * 4 - 2]     # test well outside the data too
        lerr = max(lerr, abs(rbf.evaluate(q) - lin(q)))
    check("thin-plate reproduces linear functions exactly", lerr < 1e-8, f"{lerr:.2e}")

    # ---- 4. multiquadric with polynomial augmentation also reproduces linear ------------
    rbf = R.RBFInterpolator(pts, vlin, "multiquadric", epsilon=1.0, polynomial=True)
    lerr2 = max(abs(rbf.evaluate([rnd() * 2 - 1, rnd() * 2 - 1]) - lin([0, 0])) for _ in range(1))
    # proper check
    lerr2 = 0.0
    for _ in range(200):
        q = [rnd() * 2 - 1, rnd() * 2 - 1]
        lerr2 = max(lerr2, abs(rbf.evaluate(q) - lin(q)))
    check("multiquadric + polynomial reproduces linear", lerr2 < 1e-7, f"{lerr2:.2e}")

    # ---- 5. works in 1-D --------------------------------------------------------------
    xs = [[rnd() * 6 - 3] for _ in range(20)]
    g = lambda p: math.sin(p[0])
    rbf1 = R.RBFInterpolator(xs, [g(p) for p in xs], "gaussian", epsilon=0.8)
    n1 = max(abs(rbf1.evaluate(xs[i]) - g(xs[i])) for i in range(len(xs)))
    check("1-D node reproduction", n1 < 1e-8, f"{n1:.2e}")
    i1 = max(abs(rbf1.evaluate([rnd() * 4 - 2]) - g([rnd() * 0])) for _ in range(1))
    i1 = 0.0
    for _ in range(200):
        q = [rnd() * 4 - 2]
        i1 = max(i1, abs(rbf1.evaluate(q) - g(q)))
    check("1-D recovers sin between nodes", i1 < 5e-2, f"{i1:.2e}")

    # ---- 6. works in 3-D --------------------------------------------------------------
    p3 = [[rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1] for _ in range(60)]
    h = lambda p: p[0] * p[1] + math.cos(p[2])
    rbf3 = R.RBFInterpolator(p3, [h(p) for p in p3], "multiquadric", epsilon=1.2)
    n3 = max(abs(rbf3.evaluate(p3[i]) - h(p3[i])) for i in range(len(p3)))
    check("3-D node reproduction", n3 < 1e-7, f"{n3:.2e}")

    # ---- 7. refining the sample density reduces interior error --------------------------
    def grid_error(k, kind="gaussian"):
        gp = []
        gv = []
        for i in range(k):
            for j in range(k):
                x = -1 + 2 * i / (k - 1)
                y = -1 + 2 * j / (k - 1)
                gp.append([x, y])
                gv.append(f([x, y]))
        rbf = R.RBFInterpolator(gp, gv, kind, epsilon=2.0)
        e = 0.0
        rr = _lcg(555)
        for _ in range(300):
            q = [rr() * 1.6 - 0.8, rr() * 1.6 - 0.8]
            e = max(e, abs(rbf.evaluate(q) - f(q)))
        return e

    e_coarse = grid_error(5)
    e_fine = grid_error(9)
    check("refining the grid reduces error", e_fine < e_coarse, f"coarse {e_coarse:.2e} fine {e_fine:.2e}")

    # ---- 8. constant function reproduced exactly (thin-plate's polynomial spans it) -----
    # A plain Gaussian interpolant does NOT reproduce a constant away from the nodes; the
    # polynomial augmentation (present in thin_plate) does, since a constant is a degree-0 poly.
    rbf_c = R.RBFInterpolator(pts, [7.0] * len(pts), "thin_plate")
    cerr = max(abs(rbf_c.evaluate([rnd() * 1.2 - 0.6, rnd() * 1.2 - 0.6]) - 7.0) for _ in range(100))
    check("constant data gives constant interpolant (thin-plate)", cerr < 1e-6, f"{cerr:.2e}")

    # ---- 9. interpolate() convenience matches the class ---------------------------------
    query = [[0.1, 0.2], [-0.3, 0.4]]
    conv = R.interpolate(pts, vals, query, "multiquadric", epsilon=1.5)
    rbf = R.RBFInterpolator(pts, vals, "multiquadric", epsilon=1.5)
    check("interpolate() == class evaluate()",
          all(abs(conv[i] - rbf.evaluate(query[i])) < 1e-12 for i in range(len(query))))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

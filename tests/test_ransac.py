"""Tests for RANSAC: recovers true line/circle under heavy outliers, beats OLS, iteration formula."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ransac import (  # noqa: E402
    ransac_line,
    ransac_circle,
    line_residual,
    circle_residual,
    ols_line,
    required_iterations,
    fit_line,
    fit_circle,
)


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
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state >> 8

    return nxt


def _rand(rng):
    return rng() / (1 << 24)  # in [0,1)


def main():
    # ---- 1. recover a known line under heavy outlier contamination ----------------------
    # true line y = 2x + 1. inliers near it, outliers uniform in a box.
    rng = _lcg(2024)
    successes = 0
    trials = 30
    for t in range(trials):
        pts = []
        n_in = 60
        for _ in range(n_in):
            x = _rand(rng) * 10
            noise = (_rand(rng) - 0.5) * 0.2
            pts.append((x, 2 * x + 1 + noise))
        n_out = 40  # 40% outliers
        for _ in range(n_out):
            pts.append((_rand(rng) * 10, _rand(rng) * 25))
        res = ransac_line(pts, threshold=0.5, seed=1000 + t, max_iterations=500)
        a, b, c = res["model"]
        # check the recovered line matches y = 2x + 1: evaluate residual at two true points
        r1 = line_residual((a, b, c), (0, 1))
        r2 = line_residual((a, b, c), (5, 11))
        if r1 < 0.3 and r2 < 0.3:
            successes += 1
    check("RANSAC recovers known line under 40% outliers", successes >= trials - 1,
          f"{successes}/{trials}")

    # ---- 2. RANSAC beats OLS on the same contaminated data ------------------------------
    rng = _lcg(99)
    pts = []
    for _ in range(60):
        x = _rand(rng) * 10
        pts.append((x, 2 * x + 1 + (_rand(rng) - 0.5) * 0.1))
    for _ in range(40):
        pts.append((_rand(rng) * 10, _rand(rng) * 25))
    res = ransac_line(pts, threshold=0.4, seed=7, max_iterations=500)
    ra = line_residual(res["model"], (5, 11))
    ols = ols_line(pts)
    ro = line_residual(ols, (5, 11))
    check("RANSAC closer to truth than OLS under outliers", ra < ro, f"ransac {ra:.3f} vs ols {ro:.3f}")

    # ---- 3. recovered inlier count is about the planted inlier count --------------------
    check("RANSAC inlier count ~ planted inliers (60)", 55 <= len(res["inliers"]) <= 65,
          f"{len(res['inliers'])}")

    # ---- 4. circle recovery under outliers ----------------------------------------------
    rng = _lcg(555)
    ok = 0
    for t in range(20):
        cx0, cy0, r0 = 3.0, -2.0, 5.0
        pts = []
        for _ in range(60):
            ang = _rand(rng) * 2 * math.pi
            rad = r0 + (_rand(rng) - 0.5) * 0.2
            pts.append((cx0 + rad * math.cos(ang), cy0 + rad * math.sin(ang)))
        for _ in range(30):
            pts.append(((_rand(rng) - 0.5) * 30, (_rand(rng) - 0.5) * 30))
        res = ransac_circle(pts, threshold=0.5, seed=200 + t, max_iterations=800)
        cx, cy, r = res["model"]
        if abs(cx - cx0) < 0.5 and abs(cy - cy0) < 0.5 and abs(r - r0) < 0.5:
            ok += 1
    check("RANSAC recovers known circle under 33% outliers", ok >= 18, f"{ok}/20")

    # ---- 5. required_iterations matches the closed form ---------------------------------
    # w=0.5, s=2, p=0.99 -> N = log(0.01)/log(1 - 0.25) = log(0.01)/log(0.75)
    expected = math.log(0.01) / math.log(0.75)
    check("iteration formula correct (w=.5,s=2)", abs(required_iterations(0.5, 2) - expected) < 1e-9)
    # more inliers -> fewer iterations
    check("more inliers need fewer iterations",
          required_iterations(0.9, 2) < required_iterations(0.5, 2))
    # bigger sample -> more iterations
    check("bigger sample needs more iterations",
          required_iterations(0.5, 3) > required_iterations(0.5, 2))

    # ---- 6. determinism under a seed ----------------------------------------------------
    r1 = ransac_line(pts, threshold=0.4, seed=42, max_iterations=200)
    r2 = ransac_line(pts, threshold=0.4, seed=42, max_iterations=200)
    check("deterministic under fixed seed", r1["inliers"] == r2["inliers"])

    # ---- 7. degenerate fits return None -------------------------------------------------
    check("coincident points -> no line", fit_line([(1, 1), (1, 1)]) is None)
    check("collinear points -> no circle", fit_circle([(0, 0), (1, 1), (2, 2)]) is None)

    # ---- 8. clean data (no outliers) is fit exactly -------------------------------------
    clean = [(x, 3 * x - 2) for x in range(10)]
    res = ransac_line(clean, threshold=1e-6, seed=1, max_iterations=100)
    check("clean data: all points inliers", len(res["inliers"]) == 10)
    check("clean data: line correct", line_residual(res["model"], (100, 298)) < 1e-6)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

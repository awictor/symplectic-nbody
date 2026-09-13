"""Tests for persistent H_0: n bars one infinite, deaths == MST edges, Betti0 == brute components."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from persistent_homology import (  # noqa: E402
    h0_barcode,
    finite_deaths,
    total_persistence,
    betti0_curve,
    brute_components,
    mst_edge_weights,
    pairwise_distances,
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
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    rng = _lcg(1)
    pts = [(rng() * 100, rng() * 100) for _ in range(20)]
    bars, merges = h0_barcode(pts)

    # ---- 1. exactly n bars, exactly one infinite ----------------------------------------
    check("n bars total", len(bars) == len(pts), f"{len(bars)}")
    infinite = [b for b in bars if b[1] == math.inf]
    check("exactly one infinite bar", len(infinite) == 1, f"{len(infinite)}")

    # ---- 2. all bars born at 0 ----------------------------------------------------------
    check("all bars born at 0", all(b[0] == 0.0 for b in bars))

    # ---- 3. finite deaths == MST edge weights (the key theorem) -------------------------
    deaths = finite_deaths(bars)
    mst = mst_edge_weights(pts)
    check("finite deaths == MST edge weights",
          len(deaths) == len(mst) and all(abs(deaths[i] - mst[i]) < 1e-9 for i in range(len(mst))),
          f"n_deaths={len(deaths)} n_mst={len(mst)}")

    # ---- 4. total persistence == MST weight ---------------------------------------------
    check("total persistence == MST weight", abs(total_persistence(bars) - sum(mst)) < 1e-9,
          f"{total_persistence(bars)} vs {sum(mst)}")

    # ---- 5. Betti_0 curve matches brute-force component count ---------------------------
    epsilons = [i * 5.0 for i in range(30)]
    curve = betti0_curve(bars, epsilons)
    ok = True
    for k, eps in enumerate(epsilons):
        if curve[k] != brute_components(pts, eps):
            # note: brute uses <= eps, barcode uses death > eps i.e. alive while eps < death;
            # they match because a component dies exactly at the merge distance
            ok = False
            check("Betti0 == brute components", False, f"eps={eps}: {curve[k]} vs {brute_components(pts, eps)}")
            break
    if ok:
        check("Betti0 curve == brute component count (all eps)", True)

    # ---- 6. Betti0 starts at n and ends at 1 --------------------------------------------
    check("Betti0(0) = n", betti0_curve(bars, [0.0])[0] == len(pts))
    check("Betti0(huge) = 1", betti0_curve(bars, [1e9])[0] == 1)

    # ---- 7. well-separated clusters -> that many long bars ------------------------------
    # three tight clusters far apart
    cluster_pts = []
    centers = [(0, 0), (100, 0), (50, 100)]
    r2 = _lcg(7)
    for cx, cy in centers:
        for _ in range(6):
            cluster_pts.append((cx + r2() * 3, cy + r2() * 3))
    cbars, _ = h0_barcode(cluster_pts)
    cdeaths = finite_deaths(cbars)
    # the 2 largest finite deaths (inter-cluster merges) should be much larger than the rest
    cdeaths.sort()
    long_bars = [d for d in cdeaths if d > 30]  # inter-cluster distances ~ 50-100
    check("three clusters -> 2 long finite bars + 1 infinite",
          len(long_bars) == 2, f"long finite deaths {long_bars}")

    # ---- 8. two points -> one finite bar = their distance -------------------------------
    bars2, _ = h0_barcode([(0, 0), (3, 4)])
    fd = finite_deaths(bars2)
    check("two points: finite death = distance 5", len(fd) == 1 and abs(fd[0] - 5.0) < 1e-9, f"{fd}")

    # ---- 9. single point -> one infinite bar --------------------------------------------
    bars1, _ = h0_barcode([(1, 1)])
    check("single point: one infinite bar", len(bars1) == 1 and bars1[0][1] == math.inf)

    # ---- 10. empty cloud ----------------------------------------------------------------
    check("empty cloud -> no bars", h0_barcode([])[0] == [])

    # ---- 11. collinear points: deaths are the gaps --------------------------------------
    line = [(0, 0), (1, 0), (3, 0), (6, 0)]  # gaps 1, 2, 3
    bars_l, _ = h0_barcode(line)
    check("collinear deaths = gaps [1,2,3]", finite_deaths(bars_l) == [1.0, 2.0, 3.0],
          f"{finite_deaths(bars_l)}")

    # ---- 12. merges recorded for each finite death --------------------------------------
    check("one merge per finite bar", len(merges) == len(pts) - 1)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

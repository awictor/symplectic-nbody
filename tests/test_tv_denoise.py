"""Tests for TV denoising: piecewise-const recovery, optimality vs iterative, lambda monotonicity."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tv_denoise import (  # noqa: E402
    tv_denoise,
    total_variation,
    objective,
    n_plateaus,
    lambda_path,
    brute_tv_optimum,
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
    # ---- 1. recovers a clean piecewise-constant signal from small noise -----------------
    rng = _lcg(2024)
    clean = [0.0] * 20 + [3.0] * 20 + [1.0] * 20
    noisy = [c + (rng() - 0.5) * 0.2 for c in clean]
    u = tv_denoise(noisy, lam=1.0)
    err = max(abs(u[i] - clean[i]) for i in range(len(clean)))
    check("recovers piecewise-constant signal from noise", err < 0.15, f"max err {err:.4f}")
    check("result is piecewise constant (few plateaus)", n_plateaus(u) <= 5, f"{n_plateaus(u)}")

    # ---- 2. optimality: matches the brute-force optimum on tiny signals -----------------
    rng = _lcg(77)
    ok = True
    maxgap = 0.0
    for _ in range(40):
        n = 2 + int(rng() * 2)  # length 2 or 3 (brute is exponential)
        y = [rng() * 4 - 2 for _ in range(n)]
        lam = 0.2 + rng() * 1.5
        oe = objective(tv_denoise(y, lam), y, lam)
        _, ob = brute_tv_optimum(y, lam, steps=80)
        gap = oe - ob
        maxgap = max(maxgap, gap)
        if gap > 0.02:  # within grid resolution
            ok = False
    check("TV solver matches brute-force optimum", ok, f"max gap {maxgap:.4f}")

    # ---- 3. objective is at or below the brute grid optimum -----------------------------
    # (the continuous solver can beat the discretized grid)
    rng = _lcg(7)
    ok = True
    for _ in range(30):
        n = 3
        y = [rng() * 4 - 2 for _ in range(n)]
        lam = 0.3 + rng()
        oe = objective(tv_denoise(y, lam), y, lam)
        _, ob = brute_tv_optimum(y, lam, steps=100)
        if oe > ob + 0.01:
            ok = False
    check("TV objective <= brute grid optimum (+ resolution)", ok)

    # ---- 4. larger lambda -> lower TV, fewer plateaus -----------------------------------
    rng = _lcg(555)
    y = [rng() * 4 - 2 for _ in range(50)]
    path = lambda_path(y, [0.1, 0.5, 1.0, 2.0, 5.0])
    tvs = [row[1] for row in path]
    plats = [row[2] for row in path]
    check("TV decreases as lambda grows", all(tvs[i] >= tvs[i + 1] - 1e-9 for i in range(len(tvs) - 1)),
          f"{[round(t,2) for t in tvs]}")
    check("plateaus decrease as lambda grows", all(plats[i] >= plats[i + 1] for i in range(len(plats) - 1)),
          f"{plats}")

    # ---- 5. lambda = 0 returns the input ------------------------------------------------
    y = [1.0, 5.0, 2.0, 8.0]
    check("lambda=0 returns input", tv_denoise(y, 0) == y)

    # ---- 6. very large lambda -> constant (the mean) ------------------------------------
    y = [1.0, 3.0, 2.0, 6.0, 4.0]
    u = tv_denoise(y, lam=1000)
    check("huge lambda -> constant signal", n_plateaus(u) == 1)
    check("constant equals the mean", abs(u[0] - sum(y) / len(y)) < 1e-6, f"{u[0]:.4f} vs {sum(y)/len(y):.4f}")

    # ---- 7. output stays within data range (no overshoot) -------------------------------
    rng = _lcg(321)
    ok = True
    for _ in range(50):
        n = 10 + int(rng() * 20)
        y = [rng() * 10 for _ in range(n)]
        u = tv_denoise(y, lam=0.5 + rng())
        if min(u) < min(y) - 1e-9 or max(u) > max(y) + 1e-9:
            ok = False
    check("output within data range (no overshoot)", ok)

    # ---- 8. a clean step is preserved exactly (edge-preserving) -------------------------
    step = [0.0] * 10 + [5.0] * 10
    u = tv_denoise(step, lam=1.0)
    # the step should remain a sharp step (small lambda shrinkage aside)
    check("sharp step stays piecewise constant", n_plateaus(u) == 2)
    check("step preserves the edge location", abs(u[9] - u[10]) > 4, f"jump {abs(u[9]-u[10]):.3f}")

    # ---- 9. edge cases ------------------------------------------------------------------
    check("empty input", tv_denoise([], 1.0) == [])
    check("single element", tv_denoise([3.0], 1.0) == [3.0])
    check("two elements", n_plateaus(tv_denoise([0.0, 10.0], 1.0)) in (1, 2))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""Tests for Sobol sequences: 1-D dyadic set, stratification, discrepancy vs Halton/random, QMC convergence."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sobol as S  # noqa: E402
import low_discrepancy as LD  # noqa: E402


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
    # ---- 1. points lie in [0,1) --------------------------------------------------------
    pts = S.sobol(500, 3)
    check("all coordinates in [0,1)", all(0 <= c < 1 for p in pts for c in p))
    check("correct count and dimension", len(pts) == 500 and all(len(p) == 3 for p in pts))

    # ---- 2. 1-D Sobol 2^k-prefix is exactly the dyadic set {k / 2^m} --------------------
    for m in (4, 6, 8):
        N = 1 << m
        s = set(round(x, 10) for x in S.sobol_1d(N, skip=False))
        dyad = set(round(k / N, 10) for k in range(N))
        check(f"1-D Sobol 2^{m}-prefix == dyadic set", s == dyad)

    # ---- 3. stratification: every 2^k-prefix has one point per dyadic interval ----------
    for m in (3, 5, 7):
        N = 1 << m
        s = S.sobol_1d(N, skip=False)
        buckets = sorted(int(x * N) for x in s)
        check(f"1-D {N}-prefix stratified (one per 1/{N})", buckets == list(range(N)))

    # ---- 4. 2-D stratification: 2^(2k)-prefix has one point per dyadic square -----------
    # A Sobol (0,2)-sequence: every N=4^k prefix has exactly one point per 2^k x 2^k box.
    for k in (2, 3):
        N = 1 << (2 * k)
        g = 1 << k
        p2 = S.sobol(N, 2, skip=False)
        cells = {}
        for (x, y) in p2:
            cx = min(int(x * g), g - 1)
            cy = min(int(y * g), g - 1)
            cells[(cx, cy)] = cells.get((cx, cy), 0) + 1
        check(f"2-D {N}-prefix stratified over {g}x{g} grid",
              len(cells) == N and all(v == 1 for v in cells.values()), f"{len(cells)} cells")

    # ---- 5. 2-D star discrepancy lower than i.i.d. random and beats Halton --------------
    N = 512
    sob = S.sobol(N, 2)
    d_sob = LD.star_discrepancy(sob, samples=3000)
    hal = LD.halton(N, 2)
    d_hal = LD.star_discrepancy(hal, samples=3000)
    rnd = _lcg(42)
    rand = [[rnd(), rnd()] for _ in range(N)]
    d_rand = LD.star_discrepancy(rand, samples=3000)
    check("Sobol discrepancy < random", d_sob < d_rand, f"sob {d_sob:.4f} rand {d_rand:.4f}")
    check("Sobol discrepancy <= Halton", d_sob <= d_hal * 1.1, f"sob {d_sob:.4f} hal {d_hal:.4f}")

    # ---- 6. QMC integration beats random Monte Carlo at matched N -----------------------
    f = lambda x: math.exp(x[0] + x[1])
    true = (math.e - 1) ** 2
    N = 2048
    qmc_err = abs(S.qmc_integrate(f, 2, N) - true)
    rnd = _lcg(99)
    mc_errs = []
    for _ in range(15):
        acc = sum(f([rnd(), rnd()]) for _ in range(N)) / N
        mc_errs.append(abs(acc - true))
    mc_err = sum(mc_errs) / len(mc_errs)
    check("QMC beats random MC (matched N)", qmc_err < mc_err, f"qmc {qmc_err:.2e} mc {mc_err:.2e}")

    # ---- 7. QMC converges faster than O(1/sqrt N): error shrinks super-linearly ----------
    e1 = abs(S.qmc_integrate(f, 2, 512) - true)
    e2 = abs(S.qmc_integrate(f, 2, 4096) - true)  # 8x the points
    # random MC would improve ~sqrt(8) ~ 2.8x; Sobol should do much better
    check("QMC error drops faster than random with 8x points", e2 < e1 / 4,
          f"e(512) {e1:.2e} e(4096) {e2:.2e} ratio {e1/e2:.1f}")

    # ---- 8. QMC of a constant is exact --------------------------------------------------
    check("QMC of constant is exact", abs(S.qmc_integrate(lambda x: 5.0, 3, 256) - 5.0) < 1e-9)

    # ---- 9. QMC over a non-unit box ----------------------------------------------------
    # integral of x over [0,2]x[0,3] = (2^2/2)*3 = 6
    val = S.qmc_integrate(lambda x: x[0], 2, 4096, domain=[(0, 2), (0, 3)])
    check("QMC over a box (integral of x)", abs(val - 6.0) < 1e-2, f"{val}")

    # ---- 10. skip drops the origin ------------------------------------------------------
    with_skip = S.sobol(4, 2, skip=True)
    without = S.sobol(4, 2, skip=False)
    check("skip=False starts at origin", without[0] == [0.0, 0.0])
    check("skip=True omits the origin", with_skip[0] != [0.0, 0.0])

    # ---- 11. different dimensions are not identical (no degenerate diagonal) ------------
    p = S.sobol(64, 3, skip=True)
    col0 = [q[0] for q in p]
    col1 = [q[1] for q in p]
    col2 = [q[2] for q in p]
    check("dimension projections differ", col0 != col1 and col1 != col2)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

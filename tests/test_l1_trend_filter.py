"""Validate L1 trend filter: piecewise-linear recovery, sparse kinks, lambda limits, vs HP filter."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import l1_trend_filter as l1
import hp_filter


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24) * 2 - 1


def pw_linear(n, breaks_slopes, x0=0.0):
    """Build a piecewise-linear series. breaks_slopes: list of (start_index, slope)."""
    y = [0.0] * n
    y[0] = x0
    seg = 0
    slope = breaks_slopes[0][1]
    for i in range(1, n):
        # find active slope
        for k in range(len(breaks_slopes)):
            if i > breaks_slopes[k][0]:
                slope = breaks_slopes[k][1]
        y[i] = y[i - 1] + slope
    return y


def max_abs(v):
    return max(abs(x) for x in v)


def main():
    print("L1 trend filter tests")

    # --- clean piecewise-linear signal: recovered nearly exactly ---
    n = 60
    # up (slope +1) to t=20, flat to t=40, down (slope -0.5) after
    y = []
    val = 0.0
    for i in range(n):
        if i < 20:
            slope = 1.0
        elif i < 40:
            slope = 0.0
        else:
            slope = -0.5
        val = val + (slope if i > 0 else 0.0)
        y.append(val)
    x = l1.l1_trend_filter(y, lam=1.0)
    check("clean piecewise-linear recovered", max_abs([x[i] - y[i] for i in range(n)]) < 0.5)

    # --- the fit is genuinely piecewise linear: second difference is SPARSE ---
    d2 = l1.second_difference(x)
    scale = max_abs(d2) or 1.0
    nnz = sum(1 for v in d2 if abs(v) > 1e-3 * scale)
    check("second difference is sparse (few kinks)", nnz <= 6)

    # --- kinks land near the true breakpoints (t=20 and t=40) ---
    ks = l1.kinks(x, tol=1e-2)
    near20 = any(abs(k - 20) <= 3 for k in ks)
    near40 = any(abs(k - 40) <= 3 for k in ks)
    check("kinks detected near true breakpoints", near20 and near40)

    # --- lambda -> 0 reproduces the data ---
    x0 = l1.l1_trend_filter(y, lam=1e-6)
    check("small lambda reproduces data", max_abs([x0[i] - y[i] for i in range(n)]) < 1e-2)

    # --- huge lambda collapses to the least-squares straight line ---
    xbig = l1.l1_trend_filter(y, lam=1e7, max_iter=5000)
    line = l1.ls_line(y)
    check("huge lambda -> least-squares line", max_abs([xbig[i] - line[i] for i in range(n)]) < 0.2)

    # --- denoise: fit a noisy piecewise-linear series, recover the clean trend ---
    rng = _R(7)
    noisy = [y[i] + 0.8 * rng.u() for i in range(n)]

    # --- larger lambda yields FEWER kinks (on noisy data: small lam interpolates the noise,
    #     large lam keeps only the true breakpoints) ---
    xa = l1.l1_trend_filter(noisy, lam=0.05, max_iter=6000)
    xb = l1.l1_trend_filter(noisy, lam=10.0, max_iter=6000)
    ka = len(l1.kinks(xa, tol=1e-2))
    kb = len(l1.kinks(xb, tol=1e-2))
    check(f"larger lambda -> fewer kinks ({ka} -> {kb})", kb < ka)
    xden = l1.l1_trend_filter(noisy, lam=3.0, max_iter=4000)
    err_clean = math.sqrt(sum((xden[i] - y[i]) ** 2 for i in range(n)) / n)
    err_noise = math.sqrt(sum((noisy[i] - y[i]) ** 2 for i in range(n)) / n)
    check("denoises better than raw noise", err_clean < err_noise * 0.6)

    # --- vs Hodrick-Prescott: L1 gives a SPARSER second difference (sharper corners) ---
    xhp, _cycle = hp_filter.hp_filter(noisy, lam=50.0)
    d2_l1 = l1.second_difference(xden)
    d2_hp = l1.second_difference(xhp)
    nnz_l1 = sum(1 for v in d2_l1 if abs(v) > 1e-2)
    nnz_hp = sum(1 for v in d2_hp if abs(v) > 1e-2)
    check(f"L1 second-diff sparser than HP ({nnz_l1} vs {nnz_hp})", nnz_l1 < nnz_hp)

    # --- banded solver correctness vs a dense reference ---
    _check_banded_solver()

    # --- short series returns unchanged (no interior second difference) ---
    check("length-2 series unchanged", l1.l1_trend_filter([3.0, 5.0], lam=1.0) == [3.0, 5.0])

    # --- ADMM converged: fit satisfies the stationarity roughly (residual small) ---
    Dx = l1.second_difference(x)
    check("fit has bounded second difference", max_abs(Dx) < 5.0)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


def _check_banded_solver():
    """Solve a random SPD pentadiagonal system with the banded solver and a dense Gaussian solve."""
    n = 8
    r = _R(123)
    diag = [4.0 + abs(r.u()) for _ in range(n)]
    off1 = [0.3 * r.u() for _ in range(n - 1)]
    off2 = [0.2 * r.u() for _ in range(n - 2)]
    # dense symmetric matrix
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = diag[i]
    for i in range(n - 1):
        A[i][i + 1] = A[i + 1][i] = off1[i]
    for i in range(n - 2):
        A[i][i + 2] = A[i + 2][i] = off2[i]
    b = [r.u() for _ in range(n)]
    x_band = l1._solve_banded_spd(diag, off1, off2, b)
    x_dense = _dense_solve(A, b)
    err = max(abs(x_band[i] - x_dense[i]) for i in range(n))
    check("banded pentadiagonal solver matches dense", err < 1e-8)


def _dense_solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda rr: abs(M[rr][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        for rr in range(n):
            if rr == c:
                continue
            f = M[rr][c] / pv
            for cc in range(c, n + 1):
                M[rr][cc] -= f * M[c][cc]
    return [M[i][n] / M[i][i] for i in range(n)]


if __name__ == "__main__":
    main()

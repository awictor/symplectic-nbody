"""Validate TLS: exact fit, swap-invariance vs OLS, reduced bias, orthogonality, plane fit, Deming."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import total_least_squares as tls


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def main():
    print("Total least squares tests")

    # --- exact recovery on a clean line ---
    xs = [float(i) for i in range(20)]
    m_true, b_true = 1.8, 4.2
    ys = [m_true * x + b_true for x in xs]
    fit = tls.fit_line(xs, ys)
    check("clean line slope exact", abs(fit["slope"] - m_true) < 1e-9)
    check("clean line intercept exact", abs(fit["intercept"] - b_true) < 1e-9)
    check("clean line zero orthogonal residual", fit["mse_perp"] < 1e-18)

    # --- SWAP INVARIANCE: TLS gives the same line if x,y swapped; OLS does not ---
    # Use substantial y-noise so the OLS asymmetry (which grows with noise) is unmistakable.
    rng = _R(3)
    xs = [rng.normal() * 3 for _ in range(200)]
    ys = [2.0 * x + 1.0 + rng.normal() * 2.5 for x in xs]  # heavy error in y
    f_xy = tls.fit_line(xs, ys)
    f_yx = tls.fit_line(ys, xs)
    # the swapped fit's slope should be the reciprocal of the original (same geometric line)
    slope_from_swap = 1.0 / f_yx["slope"]
    check("TLS swap-invariant (line unchanged)", abs(f_xy["slope"] - slope_from_swap) < 1e-6)
    # OLS is NOT swap-invariant
    m_ols_xy, _ = tls.ols(xs, ys)
    m_ols_yx, _ = tls.ols(ys, xs)
    check("OLS NOT swap-invariant", abs(m_ols_xy - 1.0 / m_ols_yx) > 0.05)

    # --- reduced bias: with error in BOTH variables, TLS beats OLS on slope ---
    m_true = 2.0
    rng = _R(11)
    n = 200
    x_clean = [rng.normal() * 4 for _ in range(n)]
    # large x-error drives strong attenuation (regression dilution grows with var(err_x))
    xs = [x_clean[i] + rng.normal() * 2.0 for i in range(n)]   # x measured with heavy error
    ys = [m_true * x_clean[i] + rng.normal() * 1.0 for i in range(n)]  # y with error
    m_tls = tls.fit_line(xs, ys)["slope"]
    m_ols, _ = tls.ols(xs, ys)
    check(f"TLS less biased than OLS (TLS {m_tls:.3f}, OLS {m_ols:.3f}, true 2.0)",
          abs(m_tls - m_true) < abs(m_ols - m_true))
    check("OLS shows attenuation (slope pulled toward 0)", m_ols < m_true - 0.1)

    # --- fitted normal is orthogonal to the direction of maximum variance ---
    fit = tls.fit_line(xs, ys)
    dx, dy = fit["direction"]
    nx, ny = fit["normal"]
    dot = dx * nx + dy * ny
    check("normal orthogonal to line direction", abs(dot) < 1e-9)
    check("normal is unit length", abs(math.sqrt(nx * nx + ny * ny) - 1.0) < 1e-9)

    # --- smallest eigenvalue == mean squared perpendicular residual ---
    resids = [tls.orthogonal_residual(fit, xs[i], ys[i]) for i in range(n)]
    mse_direct = sum(r * r for r in resids) / n
    check("mse_perp equals mean squared orthogonal residual",
          abs(fit["mse_perp"] - mse_direct) < 1e-9)

    # --- vertical line: OLS slope is infinite, TLS handles it ---
    xs = [1.0 + 1e-9 * i for i in range(20)]   # essentially constant x
    ys = [float(i) for i in range(20)]
    fit = tls.fit_line(xs, ys)
    check("vertical line: TLS normal is ~horizontal",
          abs(fit["normal"][1]) < 1e-3)  # normal points in x-direction

    # --- 3-D plane fit: recover a known plane z = 0.5x - 0.3y + 2 ---
    rng = _R(7)
    pts = []
    for _ in range(100):
        px = rng.normal() * 2
        py = rng.normal() * 2
        pz = 0.5 * px - 0.3 * py + 2.0
        pts.append([px, py, pz])
    normal, offset = tls.fit_hyperplane(pts)
    # true plane normal proportional to (0.5, -0.3, -1); normalize
    tn = [0.5, -0.3, -1.0]
    tnl = math.sqrt(sum(v * v for v in tn))
    tn = [v / tnl for v in tn]
    # normals may differ by sign; align
    if sum(normal[i] * tn[i] for i in range(3)) < 0:
        normal = [-v for v in normal]
    align = max(abs(normal[i] - tn[i]) for i in range(3))
    check("3-D plane normal recovered", align < 1e-6)

    # --- Deming with ratio=1 matches orthogonal TLS ---
    rng = _R(21)
    xs = [rng.normal() * 3 for _ in range(100)]
    ys = [1.5 * xs[i] + 2.0 + rng.normal() * 0.5 for i in range(100)]
    m_dem, b_dem = tls.deming(xs, ys, ratio=1.0)
    m_tls = tls.fit_line(xs, ys)["slope"]
    check(f"Deming(ratio=1) == TLS slope ({m_dem:.4f} vs {m_tls:.4f})", abs(m_dem - m_tls) < 1e-6)

    # --- deterministic ---
    a = tls.fit_line(xs, ys)
    c = tls.fit_line(xs, ys)
    check("deterministic", a["slope"] == c["slope"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

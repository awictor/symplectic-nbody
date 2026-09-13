"""Tests for Thomas + Crank-Nicolson: matches dense solve, analytic diffusion, stability, conservation."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thomas import (  # noqa: E402
    thomas_solve,
    dense_tridiagonal_solve,
    heat_step_explicit,
    heat_step_crank_nicolson,
    diffuse,
    total_heat,
    stability_ratio,
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
    # ---- 1. Thomas matches dense Gaussian elimination on random systems -----------------
    rng = _lcg(2024)
    maxerr = 0.0
    for _ in range(300):
        n = 2 + int(rng() * 10)
        a = [0.0] + [rng() * 2 - 1 for _ in range(n - 1)]
        c = [rng() * 2 - 1 for _ in range(n - 1)] + [0.0]
        # diagonally dominant b to guarantee a solution
        b = []
        for i in range(n):
            off = abs(a[i]) + abs(c[i])
            b.append((off + 1 + rng()) * (1 if rng() > 0.5 else -1))
        d = [rng() * 4 - 2 for _ in range(n)]
        x = thomas_solve(a, b, c, d)
        xd = dense_tridiagonal_solve(a, b, c, d)
        maxerr = max(maxerr, max(abs(x[i] - xd[i]) for i in range(n)))
    check("Thomas == dense Gaussian elimination (300 systems)", maxerr < 1e-8, f"{maxerr:.2e}")

    # ---- 2. Thomas verifies the residual A x - d = 0 ------------------------------------
    rng = _lcg(77)
    ok = True
    for _ in range(100):
        n = 3 + int(rng() * 8)
        a = [0.0] + [rng() for _ in range(n - 1)]
        c = [rng() for _ in range(n - 1)] + [0.0]
        b = [abs(a[i]) + abs(c[i]) + 2 + rng() for i in range(n)]
        d = [rng() * 4 - 2 for _ in range(n)]
        x = thomas_solve(a, b, c, d)
        for i in range(n):
            res = b[i] * x[i]
            if i > 0:
                res += a[i] * x[i - 1]
            if i < n - 1:
                res += c[i] * x[i + 1]
            if abs(res - d[i]) > 1e-9:
                ok = False
    check("Thomas solution satisfies the tridiagonal system", ok)

    # ---- 3. Crank-Nicolson diffuses a Gaussian to the analytic width --------------------
    # domain [-L, L], initial narrow Gaussian, alpha=1; width^2 grows as sigma0^2 + 2 alpha t
    L = 10.0
    N = 201
    dx = 2 * L / (N - 1)
    xs = [-L + i * dx for i in range(N)]
    alpha = 1.0
    sigma0 = 1.0
    u0 = [math.exp(-x * x / (2 * sigma0 ** 2)) for x in xs]
    dt = 0.05
    steps = 40
    t = dt * steps
    u = diffuse(u0, alpha, dt, dx, steps, scheme="crank_nicolson")
    # analytic: Gaussian of variance sigma0^2 + 2 alpha t, same total area
    var_t = sigma0 ** 2 + 2 * alpha * t
    area0 = total_heat(u0, dx)
    peak_analytic = area0 / math.sqrt(2 * math.pi * var_t)
    peak_numeric = max(u)
    check("CN Gaussian peak matches analytic broadening",
          abs(peak_numeric - peak_analytic) < 0.02 * peak_analytic,
          f"{peak_numeric:.4f} vs {peak_analytic:.4f}")

    # ---- 4. heat conserved (Dirichlet 0 far away: area decreases slowly; use reflecting check)
    # with fixed-zero boundaries far from the pulse, total heat is nearly conserved early on
    check("CN roughly conserves heat early (boundaries far)",
          abs(total_heat(u, dx) - area0) < 0.02 * area0, f"{total_heat(u, dx):.4f} vs {area0:.4f}")

    # ---- 5. sine mode decays at the exact analytic rate ---------------------------------
    # u(x,0) = sin(pi x / L') on [0, L'] with u=0 at ends; decays as exp(-alpha (pi/L')^2 t)
    Lp = 1.0
    N = 101
    dx = Lp / (N - 1)
    xs = [i * dx for i in range(N)]
    k = math.pi / Lp
    u0 = [math.sin(k * x) for x in xs]
    alpha = 0.5
    dt = 0.0005
    steps = 200
    t = dt * steps
    u = diffuse(u0, alpha, dt, dx, steps)
    # amplitude at the midpoint
    mid = N // 2
    analytic = math.sin(k * xs[mid]) * math.exp(-alpha * k * k * t)
    check("sine mode decays at exp(-alpha k^2 t) rate",
          abs(u[mid] - analytic) < 0.01 * abs(u0[mid]), f"{u[mid]:.4f} vs {analytic:.4f}")

    # ---- 6. Crank-Nicolson stable at large dt where explicit blows up -------------------
    N = 51
    dx = 1.0 / (N - 1)
    alpha = 1.0
    dt = 0.01  # r = alpha dt/dx^2 = 0.01/0.0004 = 25 >> 0.5
    r = stability_ratio(alpha, dt, dx)
    u0 = [math.sin(math.pi * i * dx) for i in range(N)]
    u_cn = diffuse(u0, alpha, dt, dx, 20, scheme="crank_nicolson")
    u_ex = diffuse(u0, alpha, dt, dx, 20, scheme="explicit")
    check("r >> 1/2 (unstable regime for explicit)", r > 0.5, f"r={r:.1f}")
    check("CN stays bounded at large dt", all(abs(v) < 2 for v in u_cn))
    check("explicit blows up at large dt", any(abs(v) > 10 for v in u_ex))

    # ---- 7. steady state: linear profile between boundaries -----------------------------
    # fixed ends 0 and 1, diffuse a long time -> linear ramp
    N = 21
    dx = 1.0 / (N - 1)
    u0 = [0.0] * N
    u0[0] = 0.0
    u0[-1] = 1.0
    u = diffuse(u0, 1.0, 0.01, dx, 2000, scheme="crank_nicolson")
    linear = [i / (N - 1) for i in range(N)]
    check("diffusion to steady state = linear profile",
          max(abs(u[i] - linear[i]) for i in range(N)) < 0.01)

    # ---- 8. edge cases ------------------------------------------------------------------
    # 2x0 + x1 = 3, x0 + 2x1 = 4  ->  x0=2/3, x1=5/3
    sol = thomas_solve([0, 1], [2, 2], [1, 0], [3, 4])
    check("2x2 tridiagonal solve", abs(sol[0] - 2 / 3) < 1e-9 and abs(sol[1] - 5 / 3) < 1e-9)
    try:
        thomas_solve([0], [0], [0], [1])  # zero pivot
        check("zero pivot raises", False)
    except ValueError:
        check("zero pivot raises", True)
    try:
        thomas_solve([0, 1], [2], [1, 0], [3, 4])  # length mismatch
        check("length mismatch raises", False)
    except ValueError:
        check("length mismatch raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

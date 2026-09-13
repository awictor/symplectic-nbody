"""Tests for iterative solvers: converge to direct solve, GS < Jacobi < iters, Poisson analytic."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iterative_solvers import (  # noqa: E402
    jacobi,
    gauss_seidel,
    sor,
    optimal_sor_omega,
    poisson_2d,
    direct_solve,
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


def _dd_system(n, rng):
    """A strictly diagonally dominant system (guarantees all three converge)."""
    A = [[rng() * 2 - 1 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        off = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = off + 1 + rng()  # dominant, positive
    b = [rng() * 4 - 2 for _ in range(n)]
    return A, b


def _close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def main():
    # ---- 1. all three converge to the direct solve on diagonally dominant systems -------
    rng = _lcg(2024)
    ok_j = ok_gs = ok_sor = True
    for _ in range(100):
        n = 2 + int(rng() * 6)
        A, b = _dd_system(n, rng)
        xd = direct_solve(A, b)
        xj, _ = jacobi(A, b)
        xg, _ = gauss_seidel(A, b)
        xs, _ = sor(A, b, 1.2)
        if not _close(xj, xd):
            ok_j = False
        if not _close(xg, xd):
            ok_gs = False
        if not _close(xs, xd):
            ok_sor = False
    check("Jacobi converges to direct solve", ok_j)
    check("Gauss-Seidel converges to direct solve", ok_gs)
    check("SOR converges to direct solve", ok_sor)

    # ---- 2. Gauss-Seidel needs fewer iterations than Jacobi -----------------------------
    rng = _lcg(77)
    gs_wins = 0
    trials = 40
    for _ in range(trials):
        n = 5 + int(rng() * 5)
        A, b = _dd_system(n, rng)
        _, itj = jacobi(A, b, tol=1e-8)
        _, itg = gauss_seidel(A, b, tol=1e-8)
        if itg <= itj:
            gs_wins += 1
    check("Gauss-Seidel <= Jacobi iterations (most systems)", gs_wins >= trials - 2,
          f"{gs_wins}/{trials}")

    # ---- 3. residual decreases monotonically (Gauss-Seidel on SPD) ----------------------
    # build an SPD system A = B^T B + I
    rng = _lcg(7)
    n = 6
    B = [[rng() * 2 - 1 for _ in range(n)] for _ in range(n)]
    A = [[sum(B[k][i] * B[k][j] for k in range(n)) + (1.0 if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    b = [rng() for _ in range(n)]
    xd = direct_solve(A, b)
    xg, _ = gauss_seidel(A, b)
    check("Gauss-Seidel solves SPD system", _close(xg, xd))

    # ---- 4. 2-D Poisson reproduces a manufactured analytic solution ---------------------
    # u(x,y) = sin(pi x) sin(pi y) satisfies -Laplacian u = 2 pi^2 sin(pi x) sin(pi y)
    def f(x, y):
        return 2 * math.pi ** 2 * math.sin(math.pi * x) * math.sin(math.pi * y)

    def u_exact(x, y):
        return math.sin(math.pi * x) * math.sin(math.pi * y)

    n = 15
    h = 1.0 / (n + 1)
    grid, iters = poisson_2d(f, n, method="sor", tol=1e-9)
    maxerr = 0.0
    for i in range(n):
        for j in range(n):
            x = (j + 1) * h
            y = (i + 1) * h
            maxerr = max(maxerr, abs(grid[i][j] - u_exact(x, y)))
    check("2-D Poisson matches analytic sin*sin (discretization error)", maxerr < 0.01,
          f"max err {maxerr:.5f}")

    # ---- 5. all three methods give the same Poisson solution ----------------------------
    n = 10
    gj, _ = poisson_2d(f, n, method="jacobi", tol=1e-8, max_iter=50000)
    gg, _ = poisson_2d(f, n, method="gauss_seidel", tol=1e-8, max_iter=50000)
    gs, _ = poisson_2d(f, n, method="sor", tol=1e-8, max_iter=50000)
    same = all(abs(gj[i][j] - gg[i][j]) < 1e-5 and abs(gg[i][j] - gs[i][j]) < 1e-5
               for i in range(n) for j in range(n))
    check("Jacobi/GS/SOR agree on Poisson solution", same)

    # ---- 6. optimal SOR beats Gauss-Seidel in iterations on Poisson ---------------------
    n = 20
    _, it_gs = poisson_2d(f, n, method="gauss_seidel", tol=1e-8, max_iter=50000)
    _, it_sor = poisson_2d(f, n, method="sor", tol=1e-8, max_iter=50000)
    check("optimal SOR fewer iterations than Gauss-Seidel on Poisson", it_sor < it_gs,
          f"SOR {it_sor} vs GS {it_gs}")
    check("Gauss-Seidel Poisson iters fewer than Jacobi",
          poisson_2d(f, n, method="gauss_seidel", tol=1e-8, max_iter=50000)[1] <
          poisson_2d(f, n, method="jacobi", tol=1e-8, max_iter=50000)[1])

    # ---- 7. optimal omega in (1, 2) -----------------------------------------------------
    om = optimal_sor_omega(20)
    check("optimal SOR omega in (1, 2)", 1 < om < 2, f"{om:.4f}")
    check("optimal omega grows toward 2 as grid refines", optimal_sor_omega(50) > optimal_sor_omega(10))

    # ---- 8. edge cases ------------------------------------------------------------------
    # 2x2 known: [[4,1],[1,3]] x = [1,2] -> x = [1/11, 7/11]
    A = [[4, 1], [1, 3]]
    b = [1, 2]
    xg, _ = gauss_seidel(A, b)
    check("2x2 Gauss-Seidel matches exact", _close(xg, [1 / 11, 7 / 11], tol=1e-6))
    try:
        sor(A, b, 2.5)
        check("omega out of range raises", False)
    except ValueError:
        check("omega out of range raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

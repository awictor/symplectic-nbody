"""Tests for gmres: solves match LU, residual monotone, restart works, preconditioner helps, matrix-free."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gmres import (gmres, gmres_restart, matvec_from_matrix, true_residual_norm, _norm)  # noqa: E402
import linsolve  # noqa: E402


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


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24) * 2 - 1


def diag_dominant(n, rng, symmetric=False):
    A = [[rng.u() for _ in range(n)] for _ in range(n)]
    if symmetric:
        for i in range(n):
            for j in range(i):
                A[i][j] = A[j][i]
    for i in range(n):
        A[i][i] += n              # ensure nonsingular / diagonally dominant
    return A


def main():
    rng = LCG(2024)

    # ---- 1. solves match a direct LU solve, nonsymmetric systems ----------------------
    for n in (5, 10, 20, 30):
        A = diag_dominant(n, rng)
        b = [rng.u() for _ in range(n)]
        mv = matvec_from_matrix(A)
        x, res, conv = gmres(mv, b, tol=1e-12)
        xref = linsolve.solve(A, b)
        maxdiff = max(abs(x[i] - xref[i]) for i in range(n))
        check(f"GMRES matches LU (nonsymmetric n={n})", conv and maxdiff < 1e-8,
              f"converged={conv} maxdiff={maxdiff:.2e}")

    # ---- 2. true residual falls below tolerance --------------------------------------
    A = diag_dominant(25, rng)
    b = [rng.u() for _ in range(25)]
    mv = matvec_from_matrix(A)
    x, res, conv = gmres(mv, b, tol=1e-10)
    check("true residual below tolerance", true_residual_norm(mv, b, x) / _norm(b) < 1e-9,
          f"{true_residual_norm(mv, b, x)/_norm(b):.2e}")

    # ---- 3. residual history is monotonically non-increasing (minimal residual) -------
    mono = all(res[i] >= res[i + 1] - 1e-12 for i in range(len(res) - 1))
    check("residual history monotonically non-increasing", mono)

    # ---- 4. full GMRES converges within n iterations ----------------------------------
    check("converges within n iterations", len(res) - 1 <= 25)

    # ---- 5. symmetric system: GMRES agrees with the answer (and CG-style SPD) ---------
    A = diag_dominant(15, rng, symmetric=True)
    # make SPD: A^T A is SPD but let's just use diag-dominant symmetric (SPD here)
    b = [rng.u() for _ in range(15)]
    mv = matvec_from_matrix(A)
    x, _, conv = gmres(mv, b, tol=1e-12)
    xref = linsolve.solve(A, b)
    check("GMRES on symmetric system matches LU", conv and
          max(abs(x[i] - xref[i]) for i in range(15)) < 1e-8)

    # ---- 6. restarted GMRES(m) reaches the same solution ------------------------------
    A = diag_dominant(40, rng)
    b = [rng.u() for _ in range(40)]
    mv = matvec_from_matrix(A)
    x_full, _, _ = gmres(mv, b, tol=1e-11)
    x_r, res_r, conv_r = gmres_restart(mv, b, tol=1e-11, restart=10, max_restarts=50)
    check("restarted GMRES(10) converges", conv_r)
    check("restarted GMRES matches full GMRES",
          max(abs(x_full[i] - x_r[i]) for i in range(40)) < 1e-6,
          f"{max(abs(x_full[i]-x_r[i]) for i in range(40)):.2e}")

    # ---- 7. diagonal preconditioner reduces iterations on an ill-scaled system --------
    n = 30
    # widely varying diagonal scales -> ill-conditioned
    base = diag_dominant(n, rng)
    scales = [10 ** (rng.u() * 3) for _ in range(n)]
    A = [[base[i][j] * scales[i] for j in range(n)] for i in range(n)]
    b = [rng.u() for _ in range(n)]
    mv = matvec_from_matrix(A)
    # Jacobi (diagonal) preconditioner: M^-1 r = r / diag(A)
    diag = [A[i][i] for i in range(n)]
    def jacobi_prec(r):
        return [r[i] / diag[i] for i in range(n)]
    _, res_none, conv_none = gmres(mv, b, tol=1e-8, max_iter=n)
    _, res_prec, conv_prec = gmres(mv, b, tol=1e-8, max_iter=n, precond=jacobi_prec)
    check("preconditioned GMRES converges", conv_prec)
    check("Jacobi preconditioner reduces iteration count",
          len(res_prec) <= len(res_none), f"prec {len(res_prec)} vs none {len(res_none)}")

    # ---- 8. matrix-free: operator given only as a function ----------------------------
    # tridiagonal operator (1D Laplacian + drift, nonsymmetric)
    N = 50
    def op(x):
        out = [0.0] * N
        for i in range(N):
            out[i] = 2 * x[i]
            if i > 0:
                out[i] -= 1.2 * x[i - 1]      # asymmetric coupling
            if i < N - 1:
                out[i] -= 0.8 * x[i + 1]
        return out
    b = [1.0] * N
    x, res, conv = gmres(op, b, tol=1e-10)
    check("matrix-free operator solved", conv and true_residual_norm(op, b, x) < 1e-8,
          f"resid {true_residual_norm(op, b, x):.2e}")

    # ---- 9. already-solved system (x0 exact) returns immediately ----------------------
    A = diag_dominant(8, rng)
    b = [rng.u() for _ in range(8)]
    mv = matvec_from_matrix(A)
    xsol = linsolve.solve(A, b)
    x, res, conv = gmres(mv, b, x0=xsol, tol=1e-9)
    check("exact initial guess converges immediately", conv and len(res) == 1, f"{len(res)} iters")

    # ---- 10. zero RHS gives zero solution ---------------------------------------------
    A = diag_dominant(6, rng)
    mv = matvec_from_matrix(A)
    x, res, conv = gmres(mv, [0.0] * 6, tol=1e-10)
    check("zero RHS -> zero solution", all(abs(xi) < 1e-12 for xi in x))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

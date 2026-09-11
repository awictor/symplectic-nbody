"""Conjugate gradient: solving huge sparse SPD systems without a factorization.

For a symmetric positive-definite (SPD) matrix A, solving A x = b by LU costs O(n^3) and stores
the whole factorization -- impossible when A has millions of rows (a finite-element mesh, an
image-deblurring operator, a graph Laplacian). The conjugate gradient method (Hestenes & Stiefel,
1952) solves it with nothing but matrix-vector products: it never forms or factors A, so a sparse
A costs only O(nnz) per step and O(n) memory.

The idea is to minimize the quadratic energy f(x) = (1/2) x^T A x - b^T x, whose minimum is the
solution. Steepest descent zig-zags; CG instead picks each search direction A-CONJUGATE to all
the previous ones (p_i^T A p_j = 0), so it never undoes earlier progress. In exact arithmetic it
therefore converges in at most n steps, and in practice reaches high accuracy in far fewer -- the
error shrinks at a rate set by sqrt(kappa), the square root of the condition number, which is why
PRECONDITIONING (solving M^{-1} A x = M^{-1} b with M ~ A but cheap to invert) is the whole game
for hard problems. The simple Jacobi preconditioner M = diag(A) already helps.

This module implements conjugate gradient and Jacobi-preconditioned CG using only mat-vec
products, tracks the residual history, and checks the solution against a dense LU solve and the
guaranteed <= n step count. Pure stdlib; the iterative-solver companion to the LU (linsolve) and
eigenvalue notes."""

from __future__ import annotations

import math

from linsolve import solve as _lu_solve      # dense reference for validation


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _axpy(a, x, y):
    """Return a*x + y (vector)."""
    return [a * x[i] + y[i] for i in range(len(x))]


def is_symmetric(A, tol: float = 1e-9) -> bool:
    n = len(A)
    return all(abs(A[i][j] - A[j][i]) <= tol for i in range(n) for j in range(n))


def conjugate_gradient(A, b, x0=None, tol: float = 1e-10, max_iter: int = None):
    """Solve A x = b for a symmetric positive-definite A by conjugate gradient. Returns
    (x, residual_history). Uses only matrix-vector products; stops when ||r|| <= tol * ||b||."""
    n = len(A)
    if max_iter is None:
        max_iter = n
    x = list(x0) if x0 else [0.0] * n
    r = [b[i] - _matvec(A, x)[i] for i in range(n)]      # residual b - A x
    p = r[:]
    rs_old = _dot(r, r)
    bnorm = math.sqrt(_dot(b, b)) or 1.0
    history = [math.sqrt(rs_old)]
    for _ in range(max_iter):
        if math.sqrt(rs_old) <= tol * bnorm:
            break
        Ap = _matvec(A, p)
        denom = _dot(p, Ap)
        if abs(denom) < 1e-300:
            break                                        # p is in A's null space (not SPD)
        alpha = rs_old / denom
        x = _axpy(alpha, p, x)                           # step along the conjugate direction
        r = _axpy(-alpha, Ap, r)
        rs_new = _dot(r, r)
        history.append(math.sqrt(rs_new))
        beta = rs_new / rs_old                           # keep the new direction A-conjugate
        p = _axpy(beta, p, r)
        rs_old = rs_new
    return x, history


def preconditioned_cg(A, b, x0=None, tol: float = 1e-10, max_iter: int = None):
    """Jacobi-preconditioned CG: solve with M = diag(A), which rescales the equations and speeds
    convergence on poorly scaled SPD systems. Returns (x, residual_history)."""
    n = len(A)
    if max_iter is None:
        max_iter = n
    diag = [A[i][i] for i in range(n)]
    if any(d == 0 for d in diag):
        raise ValueError("Jacobi preconditioner needs a nonzero diagonal")

    def apply_minv(v):
        return [v[i] / diag[i] for i in range(n)]

    x = list(x0) if x0 else [0.0] * n
    r = [b[i] - _matvec(A, x)[i] for i in range(n)]
    z = apply_minv(r)
    p = z[:]
    rz_old = _dot(r, z)
    bnorm = math.sqrt(_dot(b, b)) or 1.0
    history = [math.sqrt(_dot(r, r))]
    for _ in range(max_iter):
        if math.sqrt(_dot(r, r)) <= tol * bnorm:
            break
        Ap = _matvec(A, p)
        denom = _dot(p, Ap)
        if abs(denom) < 1e-300:
            break
        alpha = rz_old / denom
        x = _axpy(alpha, p, x)
        r = _axpy(-alpha, Ap, r)
        history.append(math.sqrt(_dot(r, r)))
        z = apply_minv(r)
        rz_new = _dot(r, z)
        beta = rz_new / rz_old
        p = _axpy(beta, p, z)        # new direction z + beta*p (search dir, not preconditioner)
        rz_old = rz_new
    return x, history


def solve(A, b, **kw):
    """Convenience wrapper returning just the solution vector."""
    return conjugate_gradient(A, b, **kw)[0]


def residual_norm(A, x, b):
    """||A x - b||."""
    Ax = _matvec(A, x)
    return math.sqrt(sum((Ax[i] - b[i]) ** 2 for i in range(len(b))))


def energy(A, x, b):
    """The quadratic energy f(x) = (1/2) x^T A x - b^T x that CG minimizes; its minimizer solves
    A x = b."""
    Ax = _matvec(A, x)
    return 0.5 * _dot(x, Ax) - _dot(b, x)


def spd_reference(A, b):
    """Dense LU solution, for checking CG (A assumed nonsingular)."""
    return _lu_solve(A, b)


def make_spd(n: int, seed: int = 1):
    """Build a random n x n symmetric positive-definite matrix (M M^T + n I) for tests/demos,
    using a seeded LCG."""
    state = seed

    def rnd():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24) * 2 - 1

    M = [[rnd() for _ in range(n)] for _ in range(n)]
    A = [[sum(M[i][k] * M[j][k] for k in range(n)) + (n if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    return A

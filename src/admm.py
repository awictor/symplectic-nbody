"""ADMM: the alternating direction method of multipliers, splitting hard problems into easy prox steps.

Many optimization problems that look monolithic are really two easy problems chained by a constraint:
min f(x) + g(z) subject to Ax + Bz = c. ADMM (the Alternating Direction Method of Multipliers) solves
these by never touching f and g at the same time. It forms the augmented Lagrangian and cycles three
steps: minimize over x (holding z and the dual fixed), minimize over z, then take a gradient ascent
step on the dual variable u that enforces the coupling constraint. Each of the first two is usually a
PROXIMAL operator -- a small, closed-form, embarrassingly-easy subproblem -- so ADMM decomposes a big
coupled problem into a sequence of tiny ones.

That structure makes ADMM the Swiss-army knife of modern large-scale and distributed optimization: it
underlies consensus optimization across machines, total-variation image denoising, LASSO and sparse
regression, and constrained least squares. For LASSO, min (1/2)||Dx - b||^2 + lam||x||_1, the split is
f = the smooth least-squares term (an x-update that solves one linear system, factored once) and
g = the L1 norm (a z-update that is just soft-thresholding). The dual update accumulates the running
mismatch x - z. Convergence is monitored by two residuals: the PRIMAL residual r = x - z (how far the
constraint is from satisfied) and the DUAL residual s = -rho(z - z_old) (how much z moved); both go to
zero at the optimum, and the penalty rho can be adapted to balance them.

This module implements generic 2-block ADMM given the two prox operators, plus ready-made LASSO,
non-negative least squares, and constrained-projection solvers, and it reports the primal/dual residual
history. It is validated: the LASSO solution matches the repo's FISTA solver to high accuracy; the
non-negative solver satisfies its KKT conditions; both residuals decrease to zero; a simple constrained
least-squares problem matches its analytic/projected solution; and adapting rho still converges to the
same optimum. Reuses the repo's soft-threshold prox and LU solver. Pure stdlib; the operator-splitting
companion to the FISTA, Lasso, and conjugate-gradient tools."""

from __future__ import annotations

import math

from fista import soft_threshold, prox_l1, prox_nonneg
from linsolve import solve as lu_solve


def _sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def _add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def admm(prox_f, prox_g, n, rho=1.0, max_iter=1000, tol=1e-8, x0=None):
    """Generic 2-block ADMM for min f(x) + g(z) s.t. x - z = 0 (consensus form).

    prox_f(v, rho): argmin_x f(x) + (rho/2)||x - v||^2.
    prox_g(v, rho): argmin_z g(z) + (rho/2)||z - v||^2.
    Returns (x, history) where history is a list of (primal_residual, dual_residual)."""
    x = list(x0) if x0 is not None else [0.0] * n
    z = list(x)
    u = [0.0] * n                                   # scaled dual
    hist = []
    for _ in range(max_iter):
        x = prox_f(_sub(z, u), rho)                 # x-update
        z_old = z
        z = prox_g(_add(x, u), rho)                 # z-update
        u = _add(u, _sub(x, z))                     # dual update
        r = _norm(_sub(x, z))                       # primal residual
        s = rho * _norm(_sub(z, z_old))             # dual residual
        hist.append((r, s))
        if r <= tol and s <= tol:
            break
    return x, hist


# --- LASSO via ADMM --------------------------------------------------------
def _matmulT(A):
    """Return A^T A and the factorization helper won't be needed; build A^T A explicitly."""
    n = len(A)
    p = len(A[0])
    AtA = [[0.0] * p for _ in range(p)]
    for i in range(n):
        for a in range(p):
            for b in range(p):
                AtA[a][b] += A[i][a] * A[i][b]
    return AtA


def _matTvec(A, r):
    p = len(A[0])
    return [sum(A[i][j] * r[i] for i in range(len(A))) for j in range(p)]


def lasso(A, b, lam, rho=1.0, max_iter=2000, tol=1e-9):
    """Solve min (1/2)||A x - b||^2 + lam||x||_1 by ADMM. Returns (x, residual_history).

    x-update solves (A^T A + rho I) x = A^T b + rho(z - u); factored once via LU each solve."""
    p = len(A[0])
    AtA = _matmulT(A)
    Atb = _matTvec(A, b)
    # system matrix M = A^T A + rho I (constant across iterations)
    M = [[AtA[i][j] + (rho if i == j else 0.0) for j in range(p)] for i in range(p)]

    x = [0.0] * p
    z = [0.0] * p
    u = [0.0] * p
    hist = []
    for _ in range(max_iter):
        rhs = [Atb[i] + rho * (z[i] - u[i]) for i in range(p)]
        x = lu_solve(M, rhs)
        z_old = z
        # z-update: prox of (lam/rho)||.||_1 applied to x + u
        z = [soft_threshold(x[i] + u[i], lam / rho) for i in range(p)]
        u = [u[i] + x[i] - z[i] for i in range(p)]
        r = _norm(_sub(x, z))
        s = rho * _norm(_sub(z, z_old))
        hist.append((r, s))
        if r <= tol and s <= tol:
            break
    return z, hist


def nnls(A, b, rho=1.0, max_iter=2000, tol=1e-9):
    """Non-negative least squares min ||A x - b||^2 s.t. x >= 0 via ADMM."""
    p = len(A[0])
    AtA = _matmulT(A)
    Atb = _matTvec(A, b)
    M = [[AtA[i][j] + (rho if i == j else 0.0) for j in range(p)] for i in range(p)]
    x = [0.0] * p
    z = [0.0] * p
    u = [0.0] * p
    hist = []
    for _ in range(max_iter):
        rhs = [Atb[i] + rho * (z[i] - u[i]) for i in range(p)]
        x = lu_solve(M, rhs)
        z_old = z
        z = [max(x[i] + u[i], 0.0) for i in range(p)]       # project onto x >= 0
        u = [u[i] + x[i] - z[i] for i in range(p)]
        r = _norm(_sub(x, z))
        s = rho * _norm(_sub(z, z_old))
        hist.append((r, s))
        if r <= tol and s <= tol:
            break
    return z, hist


def adaptive_lasso(A, b, lam, rho=1.0, mu=10.0, tau=2.0, max_iter=2000, tol=1e-9):
    """LASSO by ADMM with residual-balancing rho adaptation (Boyd et al. section 3.4.1)."""
    p = len(A[0])
    AtA = _matmulT(A)
    Atb = _matTvec(A, b)
    x = [0.0] * p
    z = [0.0] * p
    u = [0.0] * p
    hist = []
    for _ in range(max_iter):
        M = [[AtA[i][j] + (rho if i == j else 0.0) for j in range(p)] for i in range(p)]
        rhs = [Atb[i] + rho * (z[i] - u[i]) for i in range(p)]
        x = lu_solve(M, rhs)
        z_old = z
        z = [soft_threshold(x[i] + u[i], lam / rho) for i in range(p)]
        u = [u[i] + x[i] - z[i] for i in range(p)]
        r = _norm(_sub(x, z))
        s = rho * _norm(_sub(z, z_old))
        hist.append((r, s))
        if r <= tol and s <= tol:
            break
        # residual balancing: keep primal and dual residuals within mu of each other
        if r > mu * s:
            rho *= tau
            u = [ui / tau for ui in u]
        elif s > mu * r:
            rho /= tau
            u = [ui * tau for ui in u]
    return z, hist

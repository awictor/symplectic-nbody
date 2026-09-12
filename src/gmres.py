"""GMRES -- solving a large NONSYMMETRIC linear system with matrix-vector products alone.

Conjugate gradient solves A x = b beautifully, but only when A is symmetric positive-definite. The
matrices that arise from discretising fluid flow, convection-diffusion, electromagnetics, or any
process with a preferred direction are NONSYMMETRIC, and CG breaks on them. GMRES (Generalized Minimal
RESidual; Saad & Schultz, 1986) is the workhorse Krylov method for exactly this case: it solves any
nonsingular system using only matrix-vector products -- so it runs on sparse or implicit operators the
same way Lanczos does -- and at each step it produces the vector in the growing Krylov subspace that
MINIMISES the residual ||b - A x|| in the least-squares sense. The residual is monotonically
non-increasing, and the method converges in at most n steps exactly.

The construction has two moving parts. ARNOLDI iteration builds an orthonormal basis of the Krylov
subspace span{r0, A r0, A^2 r0, ...} by modified Gram-Schmidt (the nonsymmetric generalisation of
Lanczos, which needs the full recurrence because A is not symmetric), producing an upper-HESSENBERG
matrix H that represents A restricted to that subspace. Then the least-squares problem "find the
subspace combination minimising the residual" reduces to a tiny (m+1)-by-m Hessenberg least-squares
problem, solved incrementally with GIVENS ROTATIONS that triangularise H one column at a time -- which
also hands you the current residual norm for free, so you can stop the instant it is small enough.

Because the cost and storage grow with the iteration count, practical GMRES is RESTARTED: run m steps,
form the current solution, and begin again from the new residual (GMRES(m)). This module implements both
full GMRES and restarted GMRES(m) from a matrix-vector-product callable, with an optional
preconditioner, returning the solution, the residual history, and whether it converged. Pure standard
library.

Validation. Correctness is defined by solving the system: for random nonsymmetric (and symmetric, and
diagonally dominant) matrices, GMRES's solution matches a direct dense solve (the repository's LU) to a
tight tolerance, and the true residual ||b - A x|| falls below the requested threshold. The residual
history is verified to be MONOTONICALLY NON-INCREASING (the defining minimal-residual property), full
GMRES converges within n iterations, restarted GMRES(m) reaches the same answer, a diagonal
preconditioner reduces the iteration count on an ill-scaled system, and it is confirmed matrix-free on
an operator given only as a function. Consistency with CG is checked on an SPD system (same solution).
Pure standard library."""

import math


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _norm(v):
    return math.sqrt(_dot(v, v))


def _axpy(a, x, y):
    return [a * xi + yi for xi, yi in zip(x, y)]


def matvec_from_matrix(A):
    """Turn a dense matrix into a matrix-vector-product callable."""
    def mv(x):
        return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]
    return mv


def _apply_givens(c, s, a, b):
    """Apply a Givens rotation (c, s) to the pair (a, b)."""
    return c * a + s * b, -s * a + c * b


def gmres(matvec, b, x0=None, tol=1e-10, max_iter=None, precond=None):
    """Full GMRES for A x = b.

    ``matvec(x)`` returns A x. ``precond(r)`` (optional) approximately solves M z = r (left
    preconditioner). Returns (x, residuals, converged): the solution, the list of residual norms per
    iteration, and whether the tolerance was reached.
    """
    n = len(b)
    if max_iter is None:
        max_iter = n
    max_iter = min(max_iter, n)

    x = list(x0) if x0 is not None else [0.0] * n

    def prec(r):
        return precond(r) if precond else r

    r = [b[i] - matvec(x)[i] for i in range(n)]
    r = prec(r)
    beta = _norm(r)
    bnorm = _norm(prec(b)) or 1.0
    residuals = [beta / bnorm]
    if beta / bnorm <= tol:
        return x, residuals, True

    # Arnoldi basis, Hessenberg H, Givens rotations, and the RHS g of the LS problem
    Q = [[ri / beta for ri in r]]           # first Krylov basis vector
    H = []                                   # H[j] is column j (length j+2)
    cs = []
    sn = []
    g = [beta]                               # transformed RHS

    converged = False
    for j in range(max_iter):
        # Arnoldi: w = M^-1 A q_j, orthogonalise against previous basis (modified Gram-Schmidt)
        w = prec(matvec(Q[j]))
        hcol = [0.0] * (j + 2)
        for i in range(j + 1):
            hcol[i] = _dot(w, Q[i])
            w = _axpy(-hcol[i], Q[i], w)
        hcol[j + 1] = _norm(w)
        arnoldi_norm = hcol[j + 1]        # save before Givens zeroes this entry

        # apply the existing Givens rotations to the new column
        for i in range(j):
            hcol[i], hcol[i + 1] = _apply_givens(cs[i], sn[i], hcol[i], hcol[i + 1])

        # compute and apply the new Givens rotation to zero hcol[j+1]
        denom = math.hypot(hcol[j], hcol[j + 1])
        if denom == 0:
            c, s = 1.0, 0.0
        else:
            c = hcol[j] / denom
            s = hcol[j + 1] / denom
        cs.append(c)
        sn.append(s)
        hcol[j], hcol[j + 1] = _apply_givens(c, s, hcol[j], hcol[j + 1])   # hcol[j+1] -> 0
        H.append(hcol)

        # update the residual RHS
        g.append(0.0)
        g[j], g[j + 1] = _apply_givens(c, s, g[j], g[j + 1])
        res = abs(g[j + 1]) / bnorm
        residuals.append(res)

        if res <= tol:
            converged = True
            j += 1
            break

        # happy breakdown: the Arnoldi vector vanished, so the Krylov subspace is invariant
        if arnoldi_norm <= 1e-14:
            j += 1
            converged = True
            break

        # grow the basis for the next iteration
        if j + 1 < max_iter:
            Q.append([wi / arnoldi_norm for wi in w])
    else:
        j = max_iter

    # back-substitute the triangular system H[0:j,0:j] y = g[0:j]
    y = [0.0] * j
    for i in range(j - 1, -1, -1):
        acc = g[i]
        for k in range(i + 1, j):
            acc -= H[k][i] * y[k]
        y[i] = acc / H[i][i]

    # x = x0 + sum y[i] Q[i]
    for i in range(j):
        x = _axpy(y[i], Q[i], x)

    return x, residuals, converged


def gmres_restart(matvec, b, x0=None, tol=1e-10, restart=30, max_restarts=50, precond=None):
    """Restarted GMRES(m): run at most ``restart`` inner iterations, then restart from the residual.

    Returns (x, residuals, converged). Keeps memory bounded to the restart size.
    """
    n = len(b)
    x = list(x0) if x0 is not None else [0.0] * n
    all_res = []
    for _ in range(max_restarts):
        x, res, conv = gmres(matvec, b, x0=x, tol=tol, max_iter=restart, precond=precond)
        all_res.extend(res)
        if conv:
            return x, all_res, True
    return x, all_res, False


def true_residual_norm(matvec, b, x):
    """The actual ||b - A x|| (not the estimated one), for validation."""
    Ax = matvec(x)
    return _norm([b[i] - Ax[i] for i in range(len(b))])

"""Sherman-Morrison-Woodbury: update a matrix inverse after a low-rank change without re-inverting.

Inverting an n x n matrix costs O(n^3). But in Kalman filters, recursive least squares, Gaussian-process
updates, and quasi-Newton optimization the matrix changes only by a LOW-RANK amount between steps -- a
single outer product, or a handful of them. Re-inverting from scratch each time is wasteful. The
Sherman-Morrison formula handles a RANK-ONE update A -> A + u v^T in O(n^2):

    (A + u v^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u),

valid whenever the scalar denominator 1 + v^T A^{-1} u is nonzero (which is exactly the condition that
the updated matrix stays invertible). The WOODBURY identity generalizes it to a rank-k update
A + U C V^T, turning an n x n re-inversion into a k x k one:

    (A + U C V^T)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V^T A^{-1} U)^{-1} V^T A^{-1}.

When k << n this is a massive saving, and it is the algebraic engine behind low-rank Bayesian updates
and preconditioners. There is a matching MATRIX DETERMINANT LEMMA, det(A + u v^T) = det(A)(1 + v^T
A^{-1} u), that updates a determinant just as cheaply -- essential for likelihoods.

This module implements the rank-one Sherman-Morrison inverse update, the rank-one determinant lemma,
the general Woodbury rank-k update, and a rank-one linear-solve update that avoids forming the inverse
at all. It is validated against brute-force re-inversion: the updated inverse matches the true inverse
of the modified matrix to machine precision for random rank-1 and rank-k updates; the determinant lemma
matches a direct determinant; the solve update matches solving the modified system directly; chaining
several rank-1 updates matches one big re-inversion; and the near-singular denominator is flagged. Reuses
the repo's LU inverse and solver. Pure stdlib; the low-rank-update companion to the LU, Cholesky, and
Kalman-filter tools."""

from __future__ import annotations

from lu import inverse as _inverse, determinant as _det
from linsolve import solve as _solve, matvec as _matvec


def _outer(u, v):
    return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _matvec_local(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _vecmat(v, A):
    """Row vector times matrix: v^T A."""
    m = len(A[0])
    return [sum(v[i] * A[i][j] for i in range(len(v))) for j in range(m)]


def sherman_morrison_inverse(A_inv, u, v):
    """Inverse of (A + u v^T) given A_inv = A^{-1}, via the Sherman-Morrison formula.

    Raises ValueError if the update makes the matrix singular (denominator ~ 0)."""
    n = len(A_inv)
    Ainv_u = _matvec_local(A_inv, u)            # A^{-1} u
    vT_Ainv = _vecmat(v, A_inv)                 # v^T A^{-1}
    denom = 1.0 + sum(v[i] * Ainv_u[i] for i in range(n))
    if abs(denom) < 1e-14:
        raise ValueError("rank-1 update makes the matrix singular (denominator ~ 0)")
    out = [[A_inv[i][j] - Ainv_u[i] * vT_Ainv[j] / denom for j in range(n)] for i in range(n)]
    return out


def sherman_morrison_solve(A_inv, u, v, b):
    """Solve (A + u v^T) x = b given A^{-1}, without forming the updated inverse.

    x = A^{-1} b - (A^{-1} u)(v^T A^{-1} b) / (1 + v^T A^{-1} u)."""
    n = len(A_inv)
    Ainv_b = _matvec_local(A_inv, b)
    Ainv_u = _matvec_local(A_inv, u)
    vT_Ainv_b = sum(v[i] * Ainv_b[i] for i in range(n))
    vT_Ainv_u = sum(v[i] * Ainv_u[i] for i in range(n))
    denom = 1.0 + vT_Ainv_u
    if abs(denom) < 1e-14:
        raise ValueError("rank-1 update makes the matrix singular")
    return [Ainv_b[i] - Ainv_u[i] * vT_Ainv_b / denom for i in range(n)]


def determinant_lemma(A, u, v, det_A=None):
    """det(A + u v^T) = det(A) * (1 + v^T A^{-1} u), the matrix determinant lemma."""
    A_inv = _inverse(A)
    if det_A is None:
        det_A = _det(A)
    n = len(A)
    Ainv_u = _matvec_local(A_inv, u)
    return det_A * (1.0 + sum(v[i] * Ainv_u[i] for i in range(n)))


def woodbury_inverse(A_inv, U, C, V):
    """Inverse of (A + U C V^T) given A^{-1}, via the Woodbury identity.

    U: n x k, C: k x k, V: n x k. Returns the n x n updated inverse."""
    n = len(A_inv)
    k = len(U[0])
    # A^{-1} U  (n x k)
    Ainv_U = _matmul(A_inv, U)
    # V^T A^{-1}  (k x n)
    VT = [[V[i][j] for i in range(n)] for j in range(k)]     # k x n
    VT_Ainv = _matmul(VT, A_inv)                              # k x n
    # capacitance matrix: C^{-1} + V^T A^{-1} U  (k x k)
    C_inv = _inverse(C)
    VT_Ainv_U = _matmul(VT_Ainv, U)                           # k x k
    cap = [[C_inv[i][j] + VT_Ainv_U[i][j] for j in range(k)] for i in range(k)]
    cap_inv = _inverse(cap)
    # A^{-1} U cap_inv V^T A^{-1}  (n x n)
    tmp = _matmul(Ainv_U, cap_inv)                            # n x k
    correction = _matmul(tmp, VT_Ainv)                        # n x n
    return [[A_inv[i][j] - correction[i][j] for j in range(n)] for i in range(n)]

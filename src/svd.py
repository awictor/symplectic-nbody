"""Singular value decomposition and PCA: the axes a matrix acts along.

Every matrix A (m x n) factors as A = U S V^T: orthonormal directions V in the input space, an
orthonormal U in the output space, and non-negative SINGULAR VALUES s_i on the diagonal of S
saying how much A stretches each. Geometrically A sends the unit sphere to an ellipsoid, and the
s_i are its semi-axes. It is the most informative matrix factorization: it gives the rank (count
of nonzero singular values), the 2-norm and condition number (largest/smallest s_i), the best
low-rank approximation (keep the top k -- the Eckart-Young theorem, the basis of image
compression), and the pseudo-inverse for least squares.

The clean route (good enough for well-conditioned small matrices) is via the symmetric
eigenproblem: the right singular vectors V are the eigenvectors of A^T A, the singular values are
the square roots of its eigenvalues, and U = A V / s. We reuse the power-iteration + deflation
eigensolver for A^T A.

PRINCIPAL COMPONENT ANALYSIS is SVD applied to mean-centred data: the top singular vectors are
the directions of greatest variance, the singular values squared (over n-1) are those variances,
and projecting onto the first few gives the optimal low-dimensional summary of the data. This
module computes the (thin) SVD, the rank / norm / condition number, the best rank-k
reconstruction, and PCA with explained-variance, and checks A = U S V^T, orthonormality, and the
variance ordering. Pure stdlib; the factorization companion to the QR and eigenvalue notes."""

from __future__ import annotations

import math

from eigen import eigenvalues_symmetric


def _T(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def svd(A, tol: float = 1e-9):
    """Thin SVD of an m x n matrix A: returns (U, S, Vt) with A = U diag(S) Vt, U (m x r) and
    V (n x r) having orthonormal columns and S the r nonzero singular values (descending), where
    r is the numerical rank. Computed via the eigendecomposition of A^T A."""
    m, n = len(A), len(A[0])
    At = _T(A)
    AtA = _matmul(At, A)                        # n x n symmetric PSD
    eigvals, eigvecs = eigenvalues_symmetric(AtA)   # descending by magnitude
    U_cols, S, V_cols = [], [], []
    for lam, v in zip(eigvals, eigvecs):
        if lam <= tol:                          # drop null-space directions
            continue
        s = math.sqrt(lam)
        # right singular vector v (already unit); left singular vector u = A v / s
        Av = _matvec(A, v)
        u = [x / s for x in Av]
        S.append(s)
        V_cols.append(v)
        U_cols.append(u)
    # assemble as matrices (columns -> rows)
    r = len(S)
    U = [[U_cols[j][i] for j in range(r)] for i in range(m)]
    Vt = [V_cols[i][:] for i in range(r)]       # rows of Vt are the right singular vectors
    return U, S, Vt


def reconstruct(U, S, Vt):
    """Rebuild A = U diag(S) Vt from its SVD factors."""
    m = len(U)
    r = len(S)
    n = len(Vt[0])
    return [[sum(U[i][k] * S[k] * Vt[k][j] for k in range(r)) for j in range(n)] for i in range(m)]


def rank(A, tol: float = 1e-9) -> int:
    """Numerical rank: the number of singular values above tol."""
    _, S, _ = svd(A, tol)
    return len(S)


def spectral_norm(A) -> float:
    """The matrix 2-norm ||A||_2 = the largest singular value."""
    _, S, _ = svd(A)
    return S[0] if S else 0.0


def condition_number(A) -> float:
    """The 2-norm condition number: largest / smallest singular value. Infinite if the matrix is
    rank-deficient (a zero singular value, i.e. fewer nonzero singular values than columns)."""
    _, S, _ = svd(A)
    n = len(A[0])
    if len(S) < n or not S:          # a dropped (zero) singular value => singular => cond = inf
        return float("inf")
    return S[0] / S[-1]


def low_rank_approx(A, k: int):
    """Best rank-k approximation of A (Eckart-Young): keep the top k singular triples. Returns
    the reconstructed matrix."""
    U, S, Vt = svd(A)
    k = min(k, len(S))
    return reconstruct([row[:k] for row in U], S[:k], Vt[:k])


# --- PCA --------------------------------------------------------------------

def _column_means(data):
    n = len(data)
    d = len(data[0])
    return [sum(data[i][j] for i in range(n)) / n for j in range(d)]


def pca(data, n_components: int = None):
    """Principal component analysis of a data matrix (rows = samples, cols = features). Returns
    (components, explained_variance, mean): the top singular vectors as component rows, the
    variance each explains, and the feature means. Data is mean-centred first."""
    n = len(data)
    d = len(data[0])
    mean = _column_means(data)
    centred = [[data[i][j] - mean[j] for j in range(d)] for i in range(n)]
    U, S, Vt = svd(centred)
    if n_components is not None:
        Vt = Vt[:n_components]
        S = S[:n_components]
    # variance along each component: s^2 / (n - 1)
    denom = (n - 1) if n > 1 else 1
    explained = [s * s / denom for s in S]
    return Vt, explained, mean


def project(data, components, mean):
    """Project mean-centred data onto the principal components -> scores (n x k)."""
    n = len(data)
    d = len(data[0])
    centred = [[data[i][j] - mean[j] for j in range(d)] for i in range(n)]
    # score[i][c] = centred_row_i . component_c
    return [[sum(centred[i][j] * components[c][j] for j in range(d)) for c in range(len(components))]
            for i in range(n)]


def explained_variance_ratio(explained):
    """Fraction of total variance each component explains."""
    total = sum(explained)
    return [e / total for e in explained] if total > 0 else [0.0] * len(explained)

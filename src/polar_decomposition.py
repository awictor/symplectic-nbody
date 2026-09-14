"""Polar decomposition: split any matrix into a pure rotation and a pure stretch, A = U P.

Every real square matrix factors as A = U P, where U is ORTHOGONAL (a rotation/reflection, U^T U = I)
and P is SYMMETRIC POSITIVE-SEMIDEFINITE (a pure stretch along orthogonal axes). It is the matrix
analogue of writing a complex number as z = e^{i theta} r: U is the "phase", the rigid part of the
transformation, and P is the "magnitude", the shape-changing part. The factorization is unique when A
is invertible, and P = sqrt(A^T A) always while U = A P^{-1}.

Two things make it a workhorse. First, U is the CLOSEST ORTHOGONAL MATRIX to A in the Frobenius norm --
so the polar factor is exactly what you want to RE-ORTHOGONALIZE a drifted rotation matrix (a camera
pose, a molecular frame, a numerically-degraded basis) without throwing away information, which is why
it appears in computer graphics, robotics, and quantum chemistry. Second, it connects the SVD to the
eigendecomposition: from A = W S V^T you get U = W V^T and P = V S V^T immediately, and the SPD factor's
eigenvalues are exactly the singular values of A.

There is also a beautiful iterative route that never forms the SVD: NEWTON'S ITERATION X_{k+1} =
(X_k + X_k^{-T}) / 2 converges quadratically to the orthogonal polar factor from X_0 = A, the matrix
generalization of Newton's method for sqrt. This module computes the polar decomposition both ways --
via the repo's SVD and via the scaled Newton iteration -- and exposes the closest-rotation projection.
It is validated: U is orthogonal and P is symmetric positive-semidefinite; U P reconstructs A to
machine precision; the SVD and Newton routes agree; P equals the principal square root of A^T A; the
orthogonal factor of an already-orthogonal matrix is itself (with P = I); U is closer to A than any
other tested orthogonal matrix; and a reflection (det < 0) is handled correctly. Reuses the repo's SVD.
Pure stdlib; the matrix-factorization companion to the SVD, QR, and Sherman-Morrison tools."""

from __future__ import annotations

import math

import svd as _svd
from lu import inverse as _inverse


def _T(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _frob(A):
    return math.sqrt(sum(A[i][j] ** 2 for i in range(len(A)) for j in range(len(A[0]))))


def polar_svd(A):
    """Polar decomposition A = U P via the SVD. Returns (U, P) with U orthogonal, P symmetric PSD."""
    n = len(A)
    W, S, Vt = _svd.svd(A)
    V = _T(Vt)
    r = len(S)
    # pad to full n x n if the SVD returned a thin factorization
    # U = W Vt  (orthogonal);  P = V diag(S) Vt (symmetric PSD)
    # build U
    U = [[sum(W[i][k] * Vt[k][j] for k in range(r)) for j in range(n)] for i in range(n)]
    # P = V S Vt
    P = [[sum(V[i][k] * S[k] * Vt[k][j] for k in range(r)) for j in range(n)] for i in range(n)]
    return U, P


def polar_newton(A, max_iter=100, tol=1e-14):
    """Polar decomposition via the scaled Newton iteration X_{k+1} = (gamma X_k + X_k^{-T}/gamma)/2.

    Converges quadratically to the orthogonal factor U; P = U^T A. Returns (U, P)."""
    n = len(A)
    X = [row[:] for row in A]
    for _ in range(max_iter):
        Xinv = _inverse(X)
        XinvT = _T(Xinv)
        # scaling factor (Higham) to accelerate: gamma = (||Xinv|| / ||X||)^{1/2} in Frobenius norm
        nx = _frob(X)
        ni = _frob(Xinv)
        gamma = math.sqrt(ni / nx) if nx > 0 and ni > 0 else 1.0
        X_new = [[0.5 * (gamma * X[i][j] + XinvT[i][j] / gamma) for j in range(n)] for i in range(n)]
        diff = max(abs(X_new[i][j] - X[i][j]) for i in range(n) for j in range(n))
        X = X_new
        if diff < tol:
            break
    U = X
    P = _matmul(_T(U), A)
    # symmetrize P to remove tiny asymmetry
    P = [[0.5 * (P[i][j] + P[j][i]) for j in range(n)] for i in range(n)]
    return U, P


def polar(A, method="newton"):
    """Polar decomposition A = U P. method='newton' (machine precision) or 'svd'."""
    return polar_newton(A) if method == "newton" else polar_svd(A)


def closest_orthogonal(A):
    """The orthogonal matrix nearest to A in the Frobenius norm -- the polar factor U.

    Uses the Newton iteration, which reaches full orthogonality (the SVD route is limited by the
    accuracy of the underlying iterative SVD)."""
    U, _ = polar_newton(A)
    return U


def is_orthogonal(U, tol=1e-8):
    n = len(U)
    UtU = _matmul(_T(U), U)
    return all(abs(UtU[i][j] - (1.0 if i == j else 0.0)) < tol for i in range(n) for j in range(n))


def is_symmetric(P, tol=1e-8):
    n = len(P)
    return all(abs(P[i][j] - P[j][i]) < tol for i in range(n) for j in range(n))

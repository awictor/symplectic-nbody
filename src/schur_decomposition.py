"""Real Schur decomposition: every matrix is orthogonally similar to a quasi-triangular one, A = Q T Q^T.

The Schur decomposition is the theoretical bedrock of dense eigenvalue computation. It says every real
square matrix A can be written A = Q T Q^T with Q ORTHOGONAL and T in REAL SCHUR FORM: upper triangular
except for 1x1 and 2x2 blocks on the diagonal. Each 1x1 block is a real eigenvalue; each 2x2 block
carries a complex-conjugate pair of eigenvalues (whose real matrix has no real eigenvalue to expose).
So the eigenvalues of A are read straight off the diagonal blocks of T -- and unlike the
eigendecomposition, the Schur form ALWAYS exists over the reals and uses only stable orthogonal
transforms, never an ill-conditioned eigenvector basis.

The algorithm is the QR ITERATION, the crown jewel of numerical linear algebra. First reduce A to upper
Hessenberg form by orthogonal similarity (cheap, one-time). Then repeatedly form the QR factorization of
a SHIFTED matrix H - mu I = QR and recombine as RQ + mu I: this is again a similarity transform, it
preserves Hessenberg structure, and with the Wilkinson shift mu chosen from the trailing 2x2 block it
converges CUBICALLY, driving the subdiagonal entries to zero one deflated block at a time. Accumulating
all the orthogonal factors gives Q, and the converged Hessenberg matrix is T.

This module computes the real Schur decomposition via Hessenberg reduction plus shifted, deflating QR
iteration, accumulating the orthogonal factor. It is validated: Q is orthogonal; T is quasi-upper-
triangular (no two consecutive nonzero subdiagonals); Q T Q^T reconstructs A to machine precision; the
eigenvalues read from T's 1x1 and 2x2 diagonal blocks match an independent eigenvalue solver (including
complex-conjugate pairs); a symmetric matrix yields a genuinely diagonal T (all real eigenvalues); and
the trace and determinant are preserved. Reuses the repo's Hessenberg reduction and Householder QR.
Pure stdlib; the eigenvalue-decomposition companion to the QR-algorithm, Hessenberg, and SVD tools."""

from __future__ import annotations

import math

from hessenberg import hessenberg as _hessenberg


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _T(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _qr_hessenberg(H):
    """QR factorization of an upper-Hessenberg matrix by Givens rotations. Returns (Q, R)."""
    n = len(H)
    R = [row[:] for row in H]
    Q = _identity(n)
    for k in range(n - 1):
        a, b = R[k][k], R[k + 1][k]
        r = math.hypot(a, b)
        if r < 1e-300:
            continue
        c, s = a / r, b / r
        # apply Givens (c,s) to rows k, k+1 of R
        for j in range(n):
            t1 = R[k][j]
            t2 = R[k + 1][j]
            R[k][j] = c * t1 + s * t2
            R[k + 1][j] = -s * t1 + c * t2
        # accumulate into Q (columns k, k+1)
        for i in range(n):
            t1 = Q[i][k]
            t2 = Q[i][k + 1]
            Q[i][k] = c * t1 + s * t2
            Q[i][k + 1] = -s * t1 + c * t2
    return Q, R


def schur(A, max_iter=2000, tol=1e-13):
    """Real Schur decomposition A = Q T Q^T. Returns (Q, T), Q orthogonal, T quasi-upper-triangular.

    Hessenberg reduction followed by shifted, deflating QR iteration. The QR step acts on the full
    matrix (Givens rotations preserve Hessenberg structure) so the A = Q T Q^T identity holds exactly;
    `p` tracks the active (undeflated) trailing index only for shift selection and convergence."""
    n = len(A)
    T, Q = _hessenberg(A)                    # T Hessenberg, A = Q T Q^T
    p = n                                    # active block is rows/cols 0..p-1
    it = 0
    while p > 1 and it < max_iter * n:
        it += 1
        q = p - 1
        # deflation: negligible subdiagonal splits off the trailing eigenvalue/block
        if abs(T[q][q - 1]) <= tol * (abs(T[q - 1][q - 1]) + abs(T[q][q])):
            T[q][q - 1] = 0.0
            p -= 1
            continue
        # a settled 2x2 trailing block (the subdiagonal above it is negligible) deflates by 2
        if q - 1 >= 1 and abs(T[q - 1][q - 2]) <= tol * (abs(T[q - 2][q - 2]) + abs(T[q - 1][q - 1])):
            T[q - 1][q - 2] = 0.0
            p -= 2
            continue
        # Wilkinson shift from the trailing 2x2 of the active block
        a, b = T[p - 2][p - 2], T[p - 2][p - 1]
        c, d = T[p - 1][p - 2], T[p - 1][p - 1]
        tr = a + d
        det = a * d - b * c
        disc = tr * tr / 4 - det
        if p == 2:
            # the whole active block is 2x2: a complex pair stays as an irreducible 2x2 block,
            # but a real pair should be driven to triangular -- keep iterating unless complex.
            if disc < 0:
                break
        if disc >= 0:
            sq = math.sqrt(disc)
            mu1, mu2 = tr / 2 + sq, tr / 2 - sq
            mu = mu1 if abs(mu1 - d) < abs(mu2 - d) else mu2
        else:
            mu = d                            # complex pair: real corner shift
        # shifted QR on the FULL matrix (Givens keep it Hessenberg): T = R Q + mu I, Q_total *= Qk
        shifted = [[T[i][j] - (mu if i == j else 0.0) for j in range(n)] for i in range(n)]
        Qk, Rk = _qr_hessenberg(shifted)
        RQ = _matmul(Rk, Qk)
        for i in range(n):
            for j in range(n):
                T[i][j] = RQ[i][j] + (mu if i == j else 0.0)
        Q[:] = _matmul(Q, Qk)
    # clean tiny sub-subdiagonal noise
    for i in range(n):
        for j in range(n):
            if i > j + 1 and abs(T[i][j]) < 1e-10:
                T[i][j] = 0.0
    return Q, T


def eigenvalues_from_schur(T, tol=1e-9):
    """Read eigenvalues off the 1x1 and 2x2 diagonal blocks of a real Schur form T."""
    n = len(T)
    eigs = []
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1][i]) > tol:
            # 2x2 block -> complex conjugate pair (or two reals)
            a, b = T[i][i], T[i][i + 1]
            c, d = T[i + 1][i], T[i + 1][i + 1]
            tr = a + d
            det = a * d - b * c
            disc = tr * tr / 4 - det
            if disc >= 0:
                sq = math.sqrt(disc)
                eigs.append(complex(tr / 2 + sq, 0.0))
                eigs.append(complex(tr / 2 - sq, 0.0))
            else:
                sq = math.sqrt(-disc)
                eigs.append(complex(tr / 2, sq))
                eigs.append(complex(tr / 2, -sq))
            i += 2
        else:
            eigs.append(complex(T[i][i], 0.0))
            i += 1
    return eigs


def is_quasi_upper_triangular(T, tol=1e-6):
    """True if T is in real Schur form: no two consecutive nonzero subdiagonal entries.

    Uses a relative tolerance -- a subdiagonal entry counts as nonzero only if it is significant
    compared to its neighbouring diagonal entries (the QR iteration drives them toward, but not
    exactly to, zero)."""
    n = len(T)
    scale = max((abs(T[i][j]) for i in range(n) for j in range(n)), default=1.0) or 1.0
    # entries strictly below the subdiagonal must be zero
    for i in range(n):
        for j in range(n):
            if i > j + 1 and abs(T[i][j]) > tol * scale:
                return False

    def sub_significant(i):
        return abs(T[i][i - 1]) > tol * (abs(T[i - 1][i - 1]) + abs(T[i][i]) + 1e-300)

    # no two consecutive significant subdiagonals (would mean a block larger than 2x2)
    for i in range(1, n - 1):
        if sub_significant(i) and sub_significant(i + 1):
            return False
    return True


def reconstruct(Q, T):
    return _matmul(_matmul(Q, T), _T(Q))

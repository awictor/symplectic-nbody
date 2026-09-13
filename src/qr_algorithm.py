"""The QR algorithm: computing every eigenvalue of a matrix by iterating a QR factorization.

The QR algorithm (Francis & Kublanovskaya, ~1961) is how essentially every numerical library actually
finds eigenvalues -- it is one of the most important algorithms of the 20th century. The idea is
almost unreasonably simple: factor the matrix A = Q R (orthogonal times upper-triangular), then
MULTIPLY THE FACTORS BACK IN THE OPPOSITE ORDER, A' = R Q. Repeat. The sequence A, A', A'', ... is a
chain of SIMILAR matrices (same eigenvalues), and under mild conditions it converges to an upper
(quasi-)triangular form whose diagonal reveals the eigenvalues. That a bare factor-and-swap loop
should drive a matrix toward triangularity is one of the small miracles of numerical linear algebra.

Two refinements make it practical, both implemented here:

  HESSENBERG REDUCTION. Before iterating, reduce A by orthogonal similarity to upper HESSENBERG form
  (zero below the first subdiagonal). This costs O(n^3) once but makes each subsequent QR step O(n^2)
  instead of O(n^3), and the Hessenberg structure is preserved by the iteration.

  WILKINSON SHIFTS with DEFLATION. Plain QR converges linearly and stalls on clustered or complex
  eigenvalues. Subtracting a SHIFT mu*I before factoring and adding it back afterwards accelerates
  convergence to cubic; the Wilkinson shift (the eigenvalue of the trailing 2x2 block nearest the
  corner) is the standard choice. When a subdiagonal entry drops below tolerance the matrix DEFLATES:
  an eigenvalue is read off and the problem shrinks. A trailing 2x2 block with complex eigenvalues is
  solved directly, so real matrices with COMPLEX-CONJUGATE eigenvalue pairs are handled without ever
  leaving real arithmetic.

This module implements Hessenberg reduction, the shifted QR iteration with deflation, and eigenvalue
extraction (real and complex). It is validated against ground truth: eigenvalues of symmetric matrices
match the repo's Jacobi eigensolver; eigenvalues of a triangular matrix are its diagonal; the computed
eigenvalues satisfy the invariants sum = trace and product = determinant; complex-conjugate pairs of a
rotation-like matrix are found exactly; and, as an independent cross-check, the eigenvalues equal the
roots of the characteristic polynomial found by the repo's Durand-Kerner solver. Pure stdlib (cmath
only for the complex 2x2 case); the general-eigenvalue companion to the Jacobi, Lanczos, and
power-iteration methods."""

from __future__ import annotations

import cmath
import math


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for k in range(m):
            a = Ai[k]
            if a:
                Bk = B[k]
                Ci = C[i]
                for j in range(p):
                    Ci[j] += a * Bk[j]
    return C


def hessenberg(A):
    """Reduce A to upper Hessenberg form by Householder similarity transforms. Returns H."""
    n = len(A)
    H = [row[:] for row in A]
    for k in range(n - 2):
        # Householder vector to zero H[k+2:, k]
        x = [H[i][k] for i in range(k + 1, n)]
        alpha = -math.copysign(math.sqrt(sum(xi * xi for xi in x)), x[0]) if x[0] != 0 else \
            -math.sqrt(sum(xi * xi for xi in x))
        if abs(alpha) < 1e-300:
            continue
        v = x[:]
        v[0] -= alpha
        vnorm = math.sqrt(sum(vi * vi for vi in v))
        if vnorm < 1e-300:
            continue
        v = [vi / vnorm for vi in v]
        # apply H <- (I - 2vv^T) H  on rows k+1..n-1
        # left multiply
        for j in range(n):
            s = sum(v[i] * H[k + 1 + i][j] for i in range(len(v)))
            for i in range(len(v)):
                H[k + 1 + i][j] -= 2 * v[i] * s
        # right multiply H <- H (I - 2vv^T)  on columns k+1..n-1
        for i in range(n):
            s = sum(H[i][k + 1 + j] * v[j] for j in range(len(v)))
            for j in range(len(v)):
                H[i][k + 1 + j] -= 2 * s * v[j]
    return H


def _qr_decompose_square(A):
    """QR of a square matrix by modified Gram-Schmidt (handles the shifted iteration matrices)."""
    n = len(A)
    # columns
    V = [[A[i][j] for i in range(n)] for j in range(n)]
    Q = [[0.0] * n for _ in range(n)]  # columns
    R = [[0.0] * n for _ in range(n)]
    for j in range(n):
        v = V[j][:]
        for i in range(j):
            R[i][j] = sum(Q[i][k] * v[k] for k in range(n))
            v = [v[k] - R[i][j] * Q[i][k] for k in range(n)]
        nrm = math.sqrt(sum(vk * vk for vk in v))
        R[j][j] = nrm
        if nrm < 1e-300:
            Q[j] = [0.0] * n
        else:
            Q[j] = [vk / nrm for vk in v]
    # Qmat[i][j] = Q_col_j[i]
    Qmat = [[Q[j][i] for j in range(n)] for i in range(n)]
    return Qmat, R


def _eig_2x2(a, b, c, d):
    """Eigenvalues of [[a,b],[c,d]] (may be complex)."""
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4 * det
    if disc >= 0:
        s = math.sqrt(disc)
        return [(tr + s) / 2, (tr - s) / 2]
    s = cmath.sqrt(disc)
    return [(tr + s) / 2, (tr - s) / 2]


def eigenvalues(A, max_iter=1000, tol=1e-11):
    """All eigenvalues of a real square matrix A via the shifted QR algorithm with deflation.

    Returns a list of eigenvalues (floats for real ones, complex for conjugate pairs), unsorted.
    """
    n = len(A)
    if n == 0:
        return []
    if n == 1:
        return [A[0][0]]

    H = hessenberg(A)
    eigs = []
    p = n  # active submatrix is H[0:p][0:p]

    iters = 0
    while p > 0 and iters < max_iter * n:
        iters += 1
        if p == 1:
            eigs.append(H[0][0])
            break
        # check for deflation at the bottom
        if abs(H[p - 1][p - 2]) < tol * (abs(H[p - 2][p - 2]) + abs(H[p - 1][p - 1]) + 1e-300):
            eigs.append(H[p - 1][p - 1])
            p -= 1
            continue
        # check for a 2x2 block deflation (complex pair or converged pair)
        if p == 2 or abs(H[p - 2][p - 3]) < tol * (abs(H[p - 3][p - 3]) + abs(H[p - 2][p - 2]) + 1e-300):
            a, b = H[p - 2][p - 2], H[p - 2][p - 1]
            c, d = H[p - 1][p - 2], H[p - 1][p - 1]
            block = _eig_2x2(a, b, c, d)
            if isinstance(block[0], complex) and abs(block[0].imag) > tol:
                eigs.extend(block)
                p -= 2
                continue
            # real pair not yet converged -> keep iterating with a shift
        # Wilkinson shift from the trailing 2x2
        a, b = H[p - 2][p - 2], H[p - 2][p - 1]
        c, d = H[p - 1][p - 2], H[p - 1][p - 1]
        delta = (a - d) / 2
        denom = delta + math.copysign(math.sqrt(delta * delta + b * c), delta) if (delta * delta + b * c) >= 0 else delta
        if abs(denom) < 1e-300:
            mu = d
        else:
            mu = d - (b * c) / denom if (delta * delta + b * c) >= 0 else d
        # QR step on the active submatrix with shift mu
        sub = [[H[i][j] - (mu if i == j else 0.0) for j in range(p)] for i in range(p)]
        Q, R = _qr_decompose_square(sub)
        RQ = _matmul(R, Q)
        for i in range(p):
            for j in range(p):
                H[i][j] = RQ[i][j] + (mu if i == j else 0.0)

    return eigs


def spectral_invariants(eigs):
    """Return (sum, product) of eigenvalues -- should equal trace and determinant."""
    s = sum(eigs)
    p = 1
    for e in eigs:
        p = p * e
    return s, p


def trace(A):
    return sum(A[i][i] for i in range(len(A)))


def characteristic_poly_roots(A):
    """Eigenvalues as roots of the characteristic polynomial (independent cross-check).

    Builds the char-poly coefficients by the Faddeev-LeVerrier algorithm, then uses Durand-Kerner.
    """
    from durand_kerner import roots
    n = len(A)
    # Faddeev-LeVerrier: c_k coefficients of det(xI - A)
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    M = [[0.0] * n for _ in range(n)]
    c = [1.0]  # leading coefficient (x^n)
    for k in range(1, n + 1):
        # M = A*M + c[k-1]*I
        AM = _matmul(A, M)
        M = [[AM[i][j] + c[k - 1] * I[i][j] for j in range(n)] for i in range(n)]
        AMk = _matmul(A, M)
        ck = -sum(AMk[i][i] for i in range(n)) / k
        c.append(ck)
    # c is [1, c1, ..., cn] for x^n + c1 x^(n-1) + ... + cn
    return roots(c)

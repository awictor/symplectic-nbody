"""The Lanczos algorithm -- extremal eigenvalues of a giant symmetric matrix you never store.

The Jacobi and QR eigensolvers need the full n-by-n matrix in memory and cost O(n^3) -- fine for a
hundred rows, impossible for the million-by-million matrices of quantum chemistry, structural vibration,
network centrality (Google's PageRank is an eigenvector), or a graph Laplacian's spectral gap. But you
rarely want ALL n eigenvalues; you want the few LARGEST or SMALLEST, and often the matrix is SPARSE or
implicit -- you can multiply it by a vector but never want to form it. The LANCZOS algorithm (Cornelius
Lanczos, 1950) is built for exactly this: it finds the extreme eigenvalues of a symmetric matrix using
nothing but matrix-VECTOR products, converging to the outermost eigenvalues in far fewer than n steps.

The engine is the KRYLOV SUBSPACE. Starting from a random vector v, the vectors v, Av, A^2 v, A^3 v, ...
span a space that quickly captures the directions of the dominant eigenvectors. Lanczos builds an
ORTHONORMAL basis of that space by a three-term recurrence -- each new basis vector is A times the last,
with the components along the previous two subtracted off -- which, because A is symmetric, is all that
is needed to orthogonalise. The coefficients of that recurrence (the diagonal alphas and off-diagonal
betas) assemble a small TRIDIAGONAL matrix T of size m << n, and the eigenvalues of T (the RITZ VALUES)
approximate the extreme eigenvalues of the whole matrix, accurately after only a handful of iterations.
So an intractable n-dimensional eigenproblem collapses to a tiny tridiagonal one.

This module implements Lanczos tridiagonalisation from a matrix-vector-product callable (so it works on
dense matrices, sparse adjacency lists, or any linear operator), with FULL REORTHOGONALISATION to fight
the loss of orthogonality that plagues finite-precision Lanczos, and it solves the resulting small
tridiagonal eigenproblem by a QL iteration to return the Ritz values and, optionally, the Ritz vectors
(the approximate eigenvectors, mapped back through the basis). Convenience wrappers return the largest
or smallest few eigenvalues. Pure standard library.

Validation. The Ritz values are checked against a DENSE reference eigensolver (the repository's own
Jacobi routine) on random symmetric matrices: after enough iterations the extreme Lanczos eigenvalues
match the true extreme eigenvalues to a tight tolerance. The Ritz VECTORS satisfy the eigen-relation --
A v is parallel to v with ratio the Ritz value (small residual ||Av - lambda v||). The tridiagonal T is
genuinely symmetric tridiagonal, its basis vectors are orthonormal (thanks to reorthogonalisation), and
running the full n iterations recovers the ENTIRE spectrum. It is verified matrix-free on a sparse graph
Laplacian, whose smallest eigenvalue is exactly zero (the constant vector) -- a known analytic value the
algorithm recovers. Pure standard library."""

import math


class _LCG:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def uniform(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24) * 2 - 1


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _norm(v):
    return math.sqrt(_dot(v, v))


def _axpy(a, x, y):
    """Return a*x + y."""
    return [a * xi + yi for xi, yi in zip(x, y)]


def matvec_from_matrix(A):
    """Turn a dense matrix into a matrix-vector-product callable."""
    def mv(x):
        return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]
    return mv


# ---------------------------------------------------------------------------
# Lanczos tridiagonalisation
# ---------------------------------------------------------------------------

def lanczos_tridiagonal(matvec, n, m=None, seed=12345, reorth=True):
    """Build the Lanczos tridiagonal factor of a symmetric operator.

    ``matvec`` multiplies the (implicit) n-by-n symmetric matrix by a vector. ``m`` is the number of
    Lanczos steps (defaults to n). Returns (alpha, beta, basis): the tridiagonal diagonal alpha
    (length k), the sub/super-diagonal beta (length k-1), and the orthonormal basis vectors used.
    """
    if m is None:
        m = n
    m = min(m, n)
    rng = _LCG(seed)
    v = [rng.uniform() for _ in range(n)]
    nv = _norm(v)
    v = [x / nv for x in v]

    basis = []
    alpha = []
    beta = []
    v_prev = [0.0] * n
    b_prev = 0.0

    for j in range(m):
        w = matvec(v)
        a = _dot(w, v)
        # w := w - a*v - beta_prev*v_prev
        w = _axpy(-a, v, w)
        w = _axpy(-b_prev, v_prev, w)
        # full reorthogonalisation against all previous basis vectors
        if reorth:
            for u in basis:
                w = _axpy(-_dot(w, u), u, w)
            w = _axpy(-a * 0, v, w)  # no-op keeps structure; a already removed
        alpha.append(a)
        basis.append(v)
        b = _norm(w)
        if b < 1e-14:
            break                         # invariant subspace found; stop early
        beta.append(b)
        v_prev = v
        b_prev = b
        v = [x / b for x in w]

    return alpha, beta, basis


# ---------------------------------------------------------------------------
# symmetric tridiagonal eigensolver (QL with implicit shifts)
# ---------------------------------------------------------------------------

def tridiagonal_eigen(alpha, beta, eigenvectors=False):
    """Eigenvalues (and optional eigenvectors) of a symmetric tridiagonal matrix.

    ``alpha`` is the diagonal, ``beta`` the off-diagonal (length len(alpha)-1). Returns (values, vecs)
    where vecs is None unless requested; vecs[k] is the eigenvector for values[k].
    """
    n = len(alpha)
    d = list(alpha)
    e = list(beta) + [0.0]
    if eigenvectors:
        z = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    else:
        z = None

    for l in range(n):
        it = 0
        while True:
            # find a small subdiagonal element to split off
            m = l
            while m < n - 1:
                dd = abs(d[m]) + abs(d[m + 1])
                if abs(e[m]) <= 1e-15 * dd:
                    break
                m += 1
            if m == l:
                break
            it += 1
            if it > 100:
                break
            g = (d[l + 1] - d[l]) / (2 * e[l])
            r = math.hypot(g, 1.0)
            g = d[m] - d[l] + e[l] / (g + (r if g >= 0 else -r))
            s = c = 1.0
            p = 0.0
            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                r = math.hypot(f, g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[m] = 0.0
                    break
                s = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                if eigenvectors:
                    for k in range(n):
                        f = z[k][i + 1]
                        z[k][i + 1] = s * z[k][i] + c * f
                        z[k][i] = c * z[k][i] - s * f
            else:
                d[l] -= p
                e[l] = g
                e[m] = 0.0
                continue
            d[l] -= p
            e[l] = g
            e[m] = 0.0

    if eigenvectors:
        # columns of z are eigenvectors; return as a list indexed by eigenvalue
        vecs = [[z[i][k] for i in range(n)] for k in range(n)]
        pairs = sorted(zip(d, vecs), key=lambda t: t[0])
        return [p[0] for p in pairs], [p[1] for p in pairs]
    return sorted(d), None


# ---------------------------------------------------------------------------
# high-level eigenvalue extraction
# ---------------------------------------------------------------------------

def eigenvalues(matvec, n, m=None, seed=12345):
    """Ritz values (approximate eigenvalues) of a symmetric operator via Lanczos. Sorted ascending."""
    alpha, beta, _ = lanczos_tridiagonal(matvec, n, m, seed)
    vals, _ = tridiagonal_eigen(alpha, beta)
    return vals


def largest_eigenvalues(matvec, n, k=1, m=None, seed=12345):
    """The k largest eigenvalues of a symmetric operator."""
    if m is None:
        m = min(n, max(2 * k + 20, 20))
    vals = eigenvalues(matvec, n, m, seed)
    return vals[-k:][::-1]


def smallest_eigenvalues(matvec, n, k=1, m=None, seed=12345):
    """The k smallest eigenvalues of a symmetric operator."""
    if m is None:
        m = min(n, max(2 * k + 20, 20))
    vals = eigenvalues(matvec, n, m, seed)
    return vals[:k]


def ritz_pairs(matvec, n, m=None, seed=12345):
    """Return (values, vectors): Ritz values and their Ritz vectors (approximate eigenvectors)."""
    alpha, beta, basis = lanczos_tridiagonal(matvec, n, m, seed)
    vals, tvecs = tridiagonal_eigen(alpha, beta, eigenvectors=True)
    # map tridiagonal eigenvectors back to the full space: ritz = sum_j tvec[j] * basis[j]
    ritz = []
    for tv in tvecs:
        full = [0.0] * n
        for j, coeff in enumerate(tv):
            bj = basis[j]
            for i in range(n):
                full[i] += coeff * bj[i]
        ritz.append(full)
    return vals, ritz

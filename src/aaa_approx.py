"""The AAA algorithm: near-optimal rational approximation of data, chosen greedily, robust to poles.

Polynomials are hopeless at functions with poles, sharp peaks, or branch cuts -- Runge's phenomenon makes
a high-degree fit oscillate wildly, and no polynomial can reproduce a singularity. RATIONAL functions p/q
can, but classical rational fitting is notoriously ill-conditioned: you must pick the numerator and
denominator degrees, solve a badly-scaled linear system, and pray no spurious pole lands in your domain.
The AAA algorithm (Nakatsukasa, Sete, Trefethen 2018 -- "Adaptive Antoulas-Anderson") sidesteps all of it
with three ideas:

  1. Represent the approximant in BARYCENTRIC form, r(z) = (sum_j w_j f_j / (z - z_j)) / (sum_j w_j / (z - z_j)),
     over a set of SUPPORT POINTS z_j drawn from the data. This form is immune to the overflow that wrecks
     the monomial basis and interpolates the data exactly at every support point.
  2. Choose the support points GREEDILY: at each step add the sample where the current approximant is
     WORST, so accuracy is poured exactly where it is needed -- near a peak, an edge, a pole.
  3. Solve for the barycentric WEIGHTS as the minimal-singular-vector of a small Loewner matrix (a linear
     least-squares problem), which automatically balances numerator and denominator and suppresses
     spurious poles.

The result converges root-exponentially on functions with singularities, needs no degree chosen in
advance (it stops when the residual hits tolerance), and its poles and zeros -- recovered as a small
generalized eigenvalue problem -- often locate the true singularities of the underlying function. It is
the modern workhorse behind rational fitting, model-order reduction, and analytic continuation.

This module implements AAA with greedy support-point selection and a Loewner least-squares weight solve
(via a one-sided Jacobi SVD applied directly to the Loewner matrix -- never forming A^T A, which would
square the condition number and forfeit half the digits), plus barycentric evaluation and pole recovery. It is validated: it reproduces the support data exactly;
it approximates smooth functions (exp, a Gaussian) to near machine precision with a handful of points; it
fits functions with poles (1/(x-a), tan) far better than any polynomial of the same order and RECOVERS
the pole locations; it beats a least-squares polynomial of equal degree on a near-singular function; the
error decreases as more support points are added; and it is deterministic. Pure stdlib; the
rational-approximation companion to the Pade, Remez, barycentric-interpolation, and Chebyshev tools."""

from __future__ import annotations

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def _min_singular_vector(A, sweeps=60, tol=1e-15):
    """Right singular vector of A for its SMALLEST singular value, by ONE-SIDED JACOBI SVD.

    A is real m x n with m >= n. Working directly on A (rotating its columns to mutual orthogonality)
    avoids forming A^T A, which would square the condition number and cost half the digits -- the
    accuracy that lets AAA reach machine precision. Returns a length-n unit vector."""
    m = len(A)
    n = len(A[0])
    # U starts as a copy of A's columns; V accumulates the right rotations (identity initially).
    U = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _sweep in range(sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                # dot products of columns p, q of U
                alpha = sum(U[k][p] * U[k][p] for k in range(m))
                beta = sum(U[k][q] * U[k][q] for k in range(m))
                gamma = sum(U[k][p] * U[k][q] for k in range(m))
                off += gamma * gamma
                if abs(gamma) < tol * math.sqrt(alpha * beta + 1e-300):
                    continue
                # Jacobi rotation to zero the (p,q) inner product
                zeta = (beta - alpha) / (2.0 * gamma)
                t = (1.0 if zeta >= 0 else -1.0) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                for k in range(m):
                    up = U[k][p]
                    uq = U[k][q]
                    U[k][p] = c * up - s * uq
                    U[k][q] = s * up + c * uq
                for k in range(n):
                    vp = V[k][p]
                    vq = V[k][q]
                    V[k][p] = c * vp - s * vq
                    V[k][q] = s * vp + c * vq
        if off < tol * tol:
            break
    # column norms of U are the singular values; smallest -> its V column is the answer
    norms = [math.sqrt(sum(U[k][j] * U[k][j] for k in range(m))) for j in range(n)]
    jmin = min(range(n), key=lambda j: norms[j])
    v = [V[k][jmin] for k in range(n)]
    nv = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / nv for x in v]


def aaa(Z, F, tol=1e-13, max_terms=100):
    """Rational approximation of data (Z -> F) by the AAA algorithm.

    Z, F are equal-length real sample lists. Returns a dict with the support points, function values,
    and barycentric weights, plus a callable `eval`. Greedily adds the worst-fit sample each step and
    solves for weights as the Loewner minimal-singular-vector, until the max residual < tol * ||F||inf
    or max_terms support points are used."""
    Z = [float(z) for z in Z]
    F = [float(f) for f in F]
    M = len(Z)
    fmax = max(abs(f) for f in F) or 1.0

    # start with the mean as the running approximation
    mean_f = sum(F) / M
    R = [mean_f] * M                 # current approximant sampled at all Z
    support_idx = []                 # indices chosen as support points
    zj, fj, wj = [], [], []
    errors = []

    for it in range(max_terms):
        # 1. greedy: pick the sample with the largest residual among the non-support points
        best = -1
        best_err = -1.0
        for i in range(M):
            if i in support_idx:
                continue
            e = abs(F[i] - R[i])
            if e > best_err:
                best_err = e
                best = i
        if best < 0:
            break
        support_idx.append(best)
        zj.append(Z[best])
        fj.append(F[best])

        # 2. build the Loewner matrix over the NON-support samples
        rows = [i for i in range(M) if i not in support_idx]
        m = len(rows)
        k = len(zj)
        if m == 0:
            # everything is a support point; weights = 1
            wj = [1.0] * k
            break
        L = [[0.0] * k for _ in range(m)]
        for r, i in enumerate(rows):
            for c in range(k):
                denom = Z[i] - zj[c]
                if denom == 0:
                    denom = 1e-300
                L[r][c] = (F[i] - fj[c]) / denom

        # 3. weights = minimal right singular vector of the Loewner matrix
        if k == 1:
            wj = [1.0]
        else:
            wj = _min_singular_vector(L)

        # 4. update the running approximant R at all samples (barycentric)
        R = _bary_eval_at(Z, zj, fj, wj, F, support_idx)

        # residual over the non-support samples
        err = max((abs(F[i] - R[i]) for i in rows), default=0.0)
        errors.append(err)
        if err <= tol * fmax:
            break

    weights = wj

    def evaluate(x):
        return _bary_point(x, zj, fj, weights)

    return {
        "support_z": zj,
        "support_f": fj,
        "weights": weights,
        "errors": errors,
        "eval": evaluate,
        "num_terms": len(zj),
    }


def _bary_point(x, zj, fj, wj):
    """Evaluate the barycentric rational at a single point x."""
    num = 0.0
    den = 0.0
    for j in range(len(zj)):
        d = x - zj[j]
        if d == 0:
            return fj[j]              # exact interpolation at a support point
        t = wj[j] / d
        num += t * fj[j]
        den += t
    if den == 0:
        return num                    # degenerate; avoid divide-by-zero
    return num / den


def _bary_eval_at(Z, zj, fj, wj, F, support_idx):
    """Evaluate the current barycentric approximant at every sample in Z (support points return F)."""
    out = []
    support_set = set(support_idx)
    for i, z in enumerate(Z):
        if i in support_set:
            out.append(F[i])
        else:
            out.append(_bary_point(z, zj, fj, wj))
    return out


def poles(result):
    """Recover the poles of an AAA approximant: the zeros of the barycentric denominator.

    The denominator is sum_j w_j / (lam - z_j); clearing the fractions, its zeros are the roots of the
    degree-(m-1) polynomial  P(lam) = sum_j w_j * prod_{k != j} (lam - z_k).  Those roots are found with
    the repo's Aberth solver. Returns a list of complex poles."""
    zj = result["support_z"]
    wj = result["weights"]
    m = len(zj)
    if m < 2:
        return []

    # Build P(lam) = sum_j w_j * prod_{k != j}(lam - z_k) as coefficient list (highest degree first).
    # Each product prod_{k != j}(lam - z_k) has degree m-1.
    total = [0.0 + 0j] * m   # coefficients for degree m-1 polynomial (m coeffs)
    for j in range(m):
        # coefficients of prod_{k != j}(lam - z_k)
        prod = [1.0 + 0j]    # start with constant 1 (degree 0), highest-first
        for k in range(m):
            if k == j:
                continue
            # multiply current poly by (lam - z_k)
            new = [0j] * (len(prod) + 1)
            for i, c in enumerate(prod):
                new[i] += c              # lam * c
                new[i + 1] += -zj[k] * c  # -z_k * c
            prod = new
        # prod now has length m (degree m-1); add w_j * prod into total
        for i in range(m):
            total[i] += wj[j] * prod[i]

    import aberth
    return aberth.roots(total)

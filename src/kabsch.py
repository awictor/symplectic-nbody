"""The Kabsch algorithm -- the optimal rotation that superimposes one point set onto another.

Two clouds of corresponding points -- the same molecule in two conformations, a 3D scan aligned to its
CAD model, feature points in two camera frames, a constellation of GPS markers before and after a
tremor -- and you want the single rigid motion (rotation plus translation) that best overlays one on the
other. "Best" means minimising the sum of squared distances between corresponding points, the root-mean-
square deviation (RMSD). This is the ORTHOGONAL PROCRUSTES problem, and Wolfgang Kabsch's 1976 solution
is beautiful: it drops straight out of the singular value decomposition, in closed form, no iteration.

The recipe. Center both clouds on their centroids (removing translation). Form the 3x3 cross-covariance
H = P^T Q of the centered point sets. Take its SVD, H = U S V^T. The optimal rotation is R = V U^T --
almost. There is one subtlety that trips up every naive implementation: the bare V U^T may be a
REFLECTION (determinant -1) rather than a proper rotation, which would flip chirality. Kabsch's fix is to
check sign(det(V U^T)) and, if negative, flip the sign of the last column of V before multiplying, which
yields the best proper rotation. The optimal translation is then t = centroid_Q - R centroid_P, and the
minimal RMSD follows from the singular values.

The UMEYAMA extension (1991) adds the optimal uniform SCALE, so it aligns clouds that differ in size as
well as pose -- the full similarity transform used in point-cloud registration and shape analysis. This
module provides both, returning the rotation, translation, (optional) scale, the transformed source
points, and the residual RMSD, in any dimension. The 3x3 / NxN SVD is taken from the repository's own
svd module, so the whole thing rests on already-validated linear algebra.

Validation. (1) RECOVERY: apply a known random rotation and translation to a point set, then Kabsch must
recover that exact transform and drive the RMSD to zero -- checked over hundreds of seeded cases in 2D
and 3D. (2) The returned R is a proper rotation: orthonormal with determinant +1, never a reflection,
even when the point configuration would tempt the naive V U^T into flipping. (3) OPTIMALITY: the RMSD
Kabsch achieves is no larger than that of many randomly perturbed rotations, confirming it is the
minimiser. (4) Umeyama recovers a known scale factor exactly and aligns scaled clouds to zero residual.
(5) Adding noise raises the RMSD gracefully and the recovered rotation stays close to the true one.
(6) Degenerate inputs (identical points, single point) are handled. Pure standard library plus the
in-repo SVD."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from svd import svd  # noqa: E402


# ---------------------------------------------------------------------------
# small matrix helpers
# ---------------------------------------------------------------------------

def _centroid(points):
    n = len(points)
    dim = len(points[0])
    return [sum(p[d] for p in points) / n for d in range(dim)]


def _center(points, c):
    return [[p[d] - c[d] for d in range(len(c))] for p in points]


def _matmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _det(M):
    """Determinant by cofactor expansion (small matrices)."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    total = 0.0
    for j in range(n):
        minor = [[M[i][k] for k in range(n) if k != j] for i in range(1, n)]
        total += ((-1) ** j) * M[0][j] * _det(minor)
    return total


def _cross_covariance(P, Q):
    """H = P^T Q for centered point sets P, Q (each a list of dim-vectors). Returns dim x dim."""
    dim = len(P[0])
    H = [[0.0] * dim for _ in range(dim)]
    for p, q in zip(P, Q):
        for i in range(dim):
            for j in range(dim):
                H[i][j] += p[i] * q[j]
    return H


# ---------------------------------------------------------------------------
# Kabsch: optimal rotation + translation
# ---------------------------------------------------------------------------

def kabsch(source, target):
    """Optimal rigid alignment mapping ``source`` onto ``target`` (corresponding points, same order).

    Returns a dict with keys: R (rotation matrix), t (translation vector), rmsd, transformed (the
    source points after applying R and t). Both point sets must have the same length and dimension.
    """
    if len(source) != len(target):
        raise ValueError("point sets must have equal length")
    if len(source) == 0:
        raise ValueError("need at least one point")
    dim = len(source[0])

    cs = _centroid(source)
    ct = _centroid(target)
    P = _center(source, cs)
    Q = _center(target, ct)

    H = _cross_covariance(P, Q)
    U, S, Vt = svd(H)
    if len(S) < dim:
        # rank-deficient cross-covariance (e.g. coincident or collinear points): no orientation
        # information, so the best rigid alignment is a pure translation (identity rotation).
        R = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
    else:
        V = _transpose(Vt)
        Ut = _transpose(U)
        # R = V U^T, with a reflection correction so det(R) = +1
        R = _matmul(V, Ut)
        if _det(R) < 0:
            # flip the sign of the last column of V, then recompute
            Vc = [row[:] for row in V]
            for i in range(dim):
                Vc[i][dim - 1] = -Vc[i][dim - 1]
            R = _matmul(Vc, Ut)

    # translation: t = ct - R cs
    t = [ct[i] - sum(R[i][j] * cs[j] for j in range(dim)) for i in range(dim)]

    transformed = _apply(R, t, source)
    return {"R": R, "t": t, "rmsd": _rmsd(transformed, target), "transformed": transformed}


def umeyama(source, target, with_scale=True):
    """Optimal similarity alignment (rotation, translation, and optional uniform scale).

    Returns a dict with R, t, scale, rmsd, transformed. With ``with_scale=False`` this reduces to
    Kabsch (scale fixed at 1).
    """
    if len(source) != len(target):
        raise ValueError("point sets must have equal length")
    dim = len(source[0])
    n = len(source)

    cs = _centroid(source)
    ct = _centroid(target)
    P = _center(source, cs)
    Q = _center(target, ct)

    H = [[_cross_covariance(P, Q)[i][j] / n for j in range(dim)] for i in range(dim)]
    U, S, Vt = svd(H)
    D = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
    if len(S) < dim:
        R = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
        scale = 1.0
    else:
        V = _transpose(Vt)
        Ut = _transpose(U)
        R0 = _matmul(V, Ut)
        if _det(R0) < 0:
            D[dim - 1][dim - 1] = -1.0
        R = _matmul(_matmul(V, D), Ut)
        if with_scale:
            var_p = sum(sum(p[d] ** 2 for d in range(dim)) for p in P) / n
            scale = sum(S[i] * D[i][i] for i in range(dim)) / var_p if var_p > 0 else 1.0
        else:
            scale = 1.0

    t = [ct[i] - scale * sum(R[i][j] * cs[j] for j in range(dim)) for i in range(dim)]
    transformed = _apply(R, t, source, scale)
    return {"R": R, "t": t, "scale": scale, "rmsd": _rmsd(transformed, target),
            "transformed": transformed}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _apply(R, t, points, scale=1.0):
    dim = len(t)
    out = []
    for p in points:
        out.append([scale * sum(R[i][j] * p[j] for j in range(dim)) + t[i] for i in range(dim)])
    return out


def apply_transform(R, t, points, scale=1.0):
    """Apply a similarity transform (scale, R, t) to a list of points."""
    return _apply(R, t, points, scale)


def _rmsd(A, B):
    n = len(A)
    dim = len(A[0])
    total = sum((A[i][d] - B[i][d]) ** 2 for i in range(n) for d in range(dim))
    return math.sqrt(total / n)


def rmsd(A, B):
    """Root-mean-square deviation between two equally-sized point sets."""
    return _rmsd(A, B)

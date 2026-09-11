"""Classical multidimensional scaling: recovering a map from a table of distances.

Given only the pairwise DISTANCES between a set of objects -- cities on a map, the dissimilarity of
survey responses, sequence-alignment scores -- multidimensional scaling reconstructs a set of
COORDINATES whose pairwise distances match. It answers "where do these points sit relative to each
other?" from distances alone, the classic use being to rebuild a map of cities from a road-distance
table. Classical (Torgerson) MDS solves it exactly in closed form via linear algebra.

The trick is DOUBLE CENTERING. From the squared-distance matrix D2 (D2_ij = d_ij^2), the matrix

    B = -1/2 * J D2 J,     J = I - (1/n) 11'   (the centering matrix)

turns squared distances into the CENTERED inner-product (Gram) matrix B = X X' of the (unknown,
centered) coordinates. Eigendecomposing B = V L V' then gives the coordinates as X = V L^{1/2}:
the top k eigenvectors scaled by the square roots of their eigenvalues are the best k-dimensional
embedding, and the eigenvalues themselves say how much "shape" each dimension carries (large =
important, near-zero = flat, negative = the distances were not perfectly Euclidean). The recovered
map is unique only up to rotation, reflection, and translation -- distances fix shape, not
orientation.

This module builds the squared-distance and double-centered matrices, extracts the embedding by
eigendecomposition (reusing the symmetric eigensolver), and reports the eigenvalue spectrum -- with
a Procrustes alignment to compare a recovered map against a known one -- verified that it recovers a
square, a line, and random point sets so their reconstructed pairwise distances match the originals,
that a truly 2-D configuration has exactly two nonzero eigenvalues, and that the stress (distance
mismatch) is essentially zero for Euclidean inputs. Pure stdlib, built on the eigen module; a
dimensionality-reduction companion to the PCA/SVD note."""

from __future__ import annotations

import math

from eigen import eigenvalues_symmetric


def distance_matrix(points):
    """Euclidean distance matrix from a list of coordinate vectors."""
    n = len(points)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(len(points[i]))))
            D[i][j] = d
            D[j][i] = d
    return D


def _double_center(D):
    """B = -1/2 J D2 J, the centered Gram matrix, from a distance matrix D."""
    n = len(D)
    D2 = [[D[i][j] ** 2 for j in range(n)] for i in range(n)]
    row_mean = [sum(D2[i]) / n for i in range(n)]
    col_mean = [sum(D2[i][j] for i in range(n)) / n for j in range(n)]
    total_mean = sum(row_mean) / n
    B = [[-0.5 * (D2[i][j] - row_mean[i] - col_mean[j] + total_mean)
          for j in range(n)] for i in range(n)]
    return B


def classical_mds(D, n_components=2):
    """Reconstruct coordinates from a distance matrix by classical (Torgerson) MDS.

    Returns (coords, eigenvalues): coords is a list of n points in n_components dimensions, and
    eigenvalues are the top-n_components eigenvalues of the centered Gram matrix (their size tells
    how much structure each axis carries; near-zero or negative means fewer real dimensions)."""
    n = len(D)
    if n == 0:
        return [], []
    B = _double_center(D)
    vals, vecs = eigenvalues_symmetric(B)     # largest-magnitude first
    # keep the largest POSITIVE eigenvalues (negative ones mean non-Euclidean distances)
    order = sorted(range(n), key=lambda i: -vals[i])
    coords = []
    kept_vals = []
    chosen = []
    for i in order:
        if len(chosen) >= n_components:
            break
        if vals[i] <= 1e-9:
            continue
        chosen.append(i)
        kept_vals.append(vals[i])
    # coordinate of point p on axis c = vec_c[p] * sqrt(eigenvalue_c)
    for p in range(n):
        row = [vecs[chosen[c]][p] * math.sqrt(kept_vals[c]) for c in range(len(chosen))]
        # pad with zeros if fewer real dimensions than requested
        row += [0.0] * (n_components - len(row))
        coords.append(row)
    # pad eigenvalue list to n_components for a stable return shape
    kept_vals += [0.0] * (n_components - len(kept_vals))
    return coords, kept_vals


def eigenvalue_spectrum(D):
    """All eigenvalues of the centered Gram matrix, largest first -- the MDS 'scree' values that
    reveal the intrinsic dimensionality (count of clearly-positive values)."""
    B = _double_center(D)
    vals, _ = eigenvalues_symmetric(B)
    return sorted(vals, reverse=True)


def stress(D, coords):
    """Raw stress: sum of squared differences between original and reconstructed distances.
    Near zero means the embedding reproduced the distances faithfully."""
    n = len(D)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(sum((coords[i][k] - coords[j][k]) ** 2 for k in range(len(coords[i]))))
            total += (D[i][j] - d) ** 2
    return total


def reconstructed_distances(coords):
    """Distance matrix of a set of reconstructed coordinates (to compare against the input)."""
    return distance_matrix(coords)


def procrustes_align(target, source):
    """Rotate/reflect/translate `source` points to best match `target` (orthogonal Procrustes),
    so an MDS reconstruction (unique only up to rigid motion) can be compared to a known map.
    Returns the aligned source coordinates."""
    n = len(target)
    d = len(target[0])
    # center both
    tc = [sum(target[i][k] for i in range(n)) / n for k in range(d)]
    sc = [sum(source[i][k] for i in range(n)) / n for k in range(d)]
    T = [[target[i][k] - tc[k] for k in range(d)] for i in range(n)]
    S = [[source[i][k] - sc[k] for k in range(d)] for i in range(n)]
    # cross-covariance M = S' T
    M = [[sum(S[i][a] * T[i][b] for i in range(n)) for b in range(d)] for a in range(d)]
    # optimal orthogonal factor from the polar decomposition R = M (M'M)^{-1/2}.
    # M'M (NOT M M'): (M'M)_ab = sum_c M[c][a] M[c][b].
    MtM = [[sum(M[c][a] * M[c][b] for c in range(d)) for b in range(d)] for a in range(d)]
    evals, evecs = eigenvalues_symmetric(MtM)
    # (M'M)^{-1/2} = V diag(1/sqrt(l)) V'  (evecs are rows -> V[a][c] = evecs[c][a])
    inv_sqrt = [[sum(evecs[c][a] * evecs[c][b] / math.sqrt(max(evals[c], 1e-12))
                     for c in range(d)) for b in range(d)] for a in range(d)]
    R = [[sum(M[a][c] * inv_sqrt[c][b] for c in range(d)) for b in range(d)] for a in range(d)]
    # apply: aligned = S R, then shift to target's center
    aligned = []
    for i in range(n):
        row = [sum(S[i][a] * R[a][b] for a in range(d)) + tc[b] for b in range(d)]
        aligned.append(row)
    return aligned

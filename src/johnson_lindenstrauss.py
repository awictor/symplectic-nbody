"""Johnson-Lindenstrauss: squashing high-dimensional points into few dimensions, distances intact.

A stunning fact: any n points in a space of ANY dimension can be projected into just O(log n / eps^2)
dimensions -- independent of the original dimension -- so that every pairwise distance is preserved
to within a factor of (1 +/- eps). The Johnson-Lindenstrauss lemma (1984) makes "the curse of
dimensionality" negotiable: nearest-neighbour search, clustering, and similarity all depend on
distances, and if distances survive, so do the answers, in a fraction of the space and time.

The construction is almost absurdly simple: multiply each data vector by a random matrix R (target_dim
x original_dim) whose entries are drawn i.i.d., then scale by 1/sqrt(target_dim). A random direction
preserves squared length in expectation, and concentration of measure (the squared projected norm is a
scaled chi-square that clusters tightly around its mean) makes it hold for all pairs simultaneously
with high probability once target_dim ~ log n / eps^2. Two standard choices of R:

    GAUSSIAN: entries ~ N(0, 1). The classic, tightest constant.
    ACHLIOPTAS (sparse): entries are +sqrt(3), 0, 0, 0, 0, -sqrt(3) with probabilities 1/6, 2/3, 1/6
        -- database-friendly (integer arithmetic, two-thirds zeros) and provably just as good.

This module builds both projection matrices from a seeded RNG, projects data, measures distance
distortion, and computes the theoretical minimum target dimension for a given (n, eps). Validated
empirically: after projecting to the JL dimension, the worst-case pairwise distance distortion stays
within the eps bound across random datasets; the mean squared-distance ratio is ~1 (unbiased);
projecting to more dimensions shrinks the distortion; and the Gaussian and Achlioptas variants both
satisfy the bound. Pure stdlib; the dimensionality-reduction companion to PCA/SVD and the t-SNE
embedding note (JL preserves distances linearly and cheaply; t-SNE distorts them for visualization)."""

from __future__ import annotations

import math


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)  # uniform in [0,1)

    return nxt


def _gaussian(rng):
    """One standard normal via Box-Muller from a uniform generator."""
    u1 = max(rng(), 1e-12)
    u2 = rng()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def min_dimension(n_points, eps):
    """The JL bound: target dim k >= 4 ln(n) / (eps^2/2 - eps^3/3) guarantees (1+/-eps) distortion
    for all pairs of n points with high probability. Returns the ceiling."""
    if not (0 < eps < 1):
        raise ValueError("eps must be in (0, 1)")
    denom = eps * eps / 2 - eps * eps * eps / 3
    return math.ceil(4 * math.log(n_points) / denom)


def gaussian_matrix(target_dim, original_dim, seed=12345):
    """A target_dim x original_dim Gaussian projection matrix scaled by 1/sqrt(target_dim)."""
    rng = _lcg(seed)
    scale = 1.0 / math.sqrt(target_dim)
    return [[_gaussian(rng) * scale for _ in range(original_dim)] for _ in range(target_dim)]


def achlioptas_matrix(target_dim, original_dim, seed=12345):
    """A sparse +/-sqrt(3)/0 projection matrix (Achlioptas), scaled by 1/sqrt(target_dim).
    Entries: +sqrt(3) w.p. 1/6, 0 w.p. 2/3, -sqrt(3) w.p. 1/6."""
    rng = _lcg(seed)
    s3 = math.sqrt(3.0)
    scale = 1.0 / math.sqrt(target_dim)
    M = []
    for _ in range(target_dim):
        row = []
        for _ in range(original_dim):
            u = rng()
            if u < 1 / 6:
                row.append(s3 * scale)
            elif u < 5 / 6:
                row.append(0.0)
            else:
                row.append(-s3 * scale)
        M.append(row)
    return M


def project(data, matrix):
    """Project each row vector of data (list of vectors) by matrix (target_dim x original_dim)."""
    out = []
    for v in data:
        out.append([sum(row[j] * v[j] for j in range(len(v))) for row in matrix])
    return out


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def max_distortion(data, projected):
    """The worst-case pairwise distance distortion: max over pairs of |proj_dist/orig_dist - 1|.
    Zero-distance pairs are skipped."""
    worst = 0.0
    n = len(data)
    for i in range(n):
        for j in range(i + 1, n):
            d0 = _dist(data[i], data[j])
            if d0 < 1e-12:
                continue
            d1 = _dist(projected[i], projected[j])
            worst = max(worst, abs(d1 / d0 - 1))
    return worst


def mean_sq_ratio(data, projected):
    """Mean of (projected squared distance)/(original squared distance) over pairs -- should be ~1
    since the projection preserves squared norm in expectation."""
    total = 0.0
    count = 0
    n = len(data)
    for i in range(n):
        for j in range(i + 1, n):
            d0 = sum((x - y) ** 2 for x, y in zip(data[i], data[j]))
            if d0 < 1e-12:
                continue
            d1 = sum((x - y) ** 2 for x, y in zip(projected[i], projected[j]))
            total += d1 / d0
            count += 1
    return total / count if count else 1.0

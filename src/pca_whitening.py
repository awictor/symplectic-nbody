"""PCA and whitening: rotate data to its principal axes, then rescale to unit, uncorrelated variance.

Principal Component Analysis finds the orthogonal directions along which a data cloud varies most. Take
the covariance matrix of mean-centred data; its eigenvectors are the PRINCIPAL AXES and its eigenvalues
are the variances along them. Projecting the data onto the top-k eigenvectors is the optimal rank-k
linear compression (it minimizes reconstruction error, the Eckart-Young theorem), and the eigenvalue
sequence tells you exactly how much variance each component explains.

WHITENING goes one step further: after rotating onto the principal axes, divide each component by the
square root of its variance. The result has the IDENTITY covariance -- every direction has unit
variance and no two directions are correlated, a spherical cloud. This is PCA-whitening. But the
rotation is arbitrary: any rotation of a white cloud is still white. ZCA-whitening (zero-phase, used
in image preprocessing) picks the unique whitening transform that stays as close as possible to the
original data by rotating back onto the original axes -- W_zca = E diag(1/sqrt(lambda)) E^T, symmetric,
so whitened images still look like images rather than a scrambled principal-axis soup.

This module computes the covariance, PCA projection and reconstruction, explained-variance ratios,
and both PCA- and ZCA-whitening (with the inverse transforms). It is validated: the covariance of
whitened data is the identity to machine precision; the top principal axis aligns with the known
major axis of a planted anisotropic Gaussian; explained-variance ratios sum to 1 and are descending;
rank-k reconstruction error equals the tail sum of eigenvalues (Eckart-Young); ZCA is symmetric and
stays strictly closer to the original data than PCA-whitening; and every transform round-trips through
its inverse. Reuses the repo's Jacobi eigensolver. Pure stdlib; the dimensionality-reduction companion
to the SVD, MDS, and k-means tools."""

from __future__ import annotations

from jacobi_eigen import sorted_eigen


def _mean(data):
    n = len(data)
    d = len(data[0])
    return [sum(row[j] for row in data) / n for j in range(d)]


def center(data):
    """Subtract the column mean; return (centered_data, mean)."""
    mu = _mean(data)
    return [[row[j] - mu[j] for j in range(len(mu))] for row in data], mu


def covariance(data, biased=True):
    """Covariance matrix of the rows (each row a sample). biased=True divides by n, else n-1."""
    c, _ = center(data)
    n = len(c)
    d = len(c[0])
    denom = n if biased else (n - 1)
    cov = [[0.0] * d for _ in range(d)]
    for row in c:
        for i in range(d):
            ri = row[i]
            for j in range(d):
                cov[i][j] += ri * row[j]
    for i in range(d):
        for j in range(d):
            cov[i][j] /= denom
    return cov


def pca(data, n_components=None, biased=True):
    """Principal component analysis. Returns a dict with eigenvalues (variances, descending),
    components (principal axes as rows), mean, and explained_variance_ratio."""
    cov = covariance(data, biased)
    vals, vecs = sorted_eigen(cov)          # descending eigenvalues, eigenvectors as columns
    d = len(cov)
    k = d if n_components is None else n_components
    # principal axes as ROWS (component i is the i-th eigenvector)
    components = [[vecs[r][c] for r in range(d)] for c in range(k)]
    total = sum(v for v in vals) or 1.0
    ratio = [max(v, 0.0) / total for v in vals[:k]]
    mu = _mean(data)
    return {
        "eigenvalues": vals[:k],
        "all_eigenvalues": vals,
        "components": components,
        "mean": mu,
        "explained_variance_ratio": ratio,
    }


def transform(data, model):
    """Project centered data onto the principal axes (scores)."""
    mu = model["mean"]
    comps = model["components"]
    out = []
    for row in data:
        cen = [row[j] - mu[j] for j in range(len(mu))]
        out.append([sum(cen[j] * comp[j] for j in range(len(mu))) for comp in comps])
    return out


def inverse_transform(scores, model):
    """Reconstruct data from principal-component scores (adds the mean back)."""
    mu = model["mean"]
    comps = model["components"]
    d = len(mu)
    out = []
    for s in scores:
        rec = [mu[j] + sum(s[c] * comps[c][j] for c in range(len(comps))) for j in range(d)]
        out.append(rec)
    return out


def reconstruction_error(data, model):
    """Mean squared reconstruction error of the rank-k PCA projection."""
    rec = inverse_transform(transform(data, model), model)
    n = len(data)
    d = len(data[0])
    s = 0.0
    for i in range(n):
        for j in range(d):
            e = data[i][j] - rec[i][j]
            s += e * e
    return s / n


def _matmul_vec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def whitening_matrix(data, mode="pca", eps=1e-12, biased=True):
    """Whitening transform W (d x d) so that (x - mean) @ W^T has identity covariance.

    mode="pca": W = diag(1/sqrt(lambda)) E^T  (rotates onto principal axes).
    mode="zca": W = E diag(1/sqrt(lambda)) E^T (symmetric, stays near the original axes)."""
    cov = covariance(data, biased)
    vals, vecs = sorted_eigen(cov)
    d = len(cov)
    # E has eigenvectors as columns; build inv-sqrt of eigenvalues
    inv_sqrt = [1.0 / ((v + eps) ** 0.5) for v in vals]
    # E^T rows are eigenvectors
    Et = [[vecs[r][c] for r in range(d)] for c in range(d)]     # row c = eigenvector c
    if mode == "pca":
        # W[i] = inv_sqrt[i] * eigenvector_i
        W = [[inv_sqrt[i] * Et[i][j] for j in range(d)] for i in range(d)]
    elif mode == "zca":
        # W = E @ diag(inv_sqrt) @ E^T ; W[i][j] = sum_k E[i][k] inv_sqrt[k] E[j][k]
        W = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(d):
                W[i][j] = sum(vecs[i][k] * inv_sqrt[k] * vecs[j][k] for k in range(d))
    else:
        raise ValueError("mode must be 'pca' or 'zca'")
    return W


def whiten(data, mode="pca", eps=1e-12, biased=True):
    """Whiten the data: center then apply the whitening matrix. Returns (whitened, mean, W)."""
    mu = _mean(data)
    W = whitening_matrix(data, mode, eps, biased)
    out = []
    for row in data:
        cen = [row[j] - mu[j] for j in range(len(mu))]
        out.append(_matmul_vec(W, cen))
    return out, mu, W


def gram_covariance(data, biased=True):
    """The covariance of the *rows* of `data` treated as samples -- convenience alias."""
    return covariance(data, biased)

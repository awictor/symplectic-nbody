"""Non-negative matrix factorization: parts-based decomposition by multiplicative updates.

Given a non-negative data matrix V (m samples by n features -- pixel intensities, word counts,
spectra), NMF finds two non-negative matrices W (m by k) and H (k by n) whose product approximates
it: V ~ W H, with a small inner dimension k. Because nothing is allowed to be negative, the
factorization is PURELY ADDITIVE -- features combine only by piling on, never by cancellation -- and
Lee and Seung's celebrated 1999 result is that this forces a PARTS-BASED representation: the columns
of W become interpretable pieces (facial features, document topics, spectral components) and each
data point is a non-negative mix of them. This is why NMF underlies topic modelling, hyperspectral
unmixing, audio source separation, and recommender systems, where PCA's signed components would be
uninterpretable.

The workhorse is Lee and Seung's MULTIPLICATIVE UPDATE rule, which minimizes the squared
reconstruction error ||V - W H||^2 while keeping every entry non-negative automatically (a positive
matrix times a ratio of non-negative matrices stays non-negative):

    H <- H * (W^T V) / (W^T W H),
    W <- W * (V H^T) / (W H H^T).

Each update provably does not increase the error (a majorize-minimize argument), so the algorithm
converges monotonically -- no step size to tune, no projection needed. This module runs NMF with
random non-negative initialization, reports the reconstruction error curve, and also implements
multiplicative updates for the KL-divergence objective used in topic models.

Validated by recovery and monotonicity: on a matrix built as a product of known non-negative factors
the reconstruction error falls to near zero and the recovered W H reproduces V; the Frobenius error
decreases monotonically every iteration (the Lee-Seung guarantee); factors stay non-negative
throughout; a higher rank fits at least as well; and the KL objective likewise decreases. Pure
stdlib; the parts-based companion to the SVD and k-means / spectral clustering tools."""

from __future__ import annotations

import math


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)

    return nxt


def _matmul(A, B):
    m, kk = len(A), len(A[0])
    n = len(B[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        Oi = out[i]
        for t in range(kk):
            a = Ai[t]
            if a == 0:
                continue
            Bt = B[t]
            for j in range(n):
                Oi[j] += a * Bt[j]
    return out


def _transpose(A):
    return [list(col) for col in zip(*A)]


def frobenius_error(V, W, H):
    """||V - W H||_F, the Frobenius norm of the reconstruction residual."""
    WH = _matmul(W, H)
    s = 0.0
    for i in range(len(V)):
        for j in range(len(V[0])):
            d = V[i][j] - WH[i][j]
            s += d * d
    return math.sqrt(s)


def nmf(V, k, iterations=200, seed=12345, eps=1e-10, track_error=False):
    """Factor a non-negative matrix V (m x n) as W (m x k) times H (k x n) by Lee-Seung
    multiplicative updates minimizing squared error. Returns (W, H) or (W, H, error_curve)."""
    m = len(V)
    n = len(V[0])
    if any(V[i][j] < 0 for i in range(m) for j in range(n)):
        raise ValueError("V must be non-negative")
    rng = _lcg(seed)
    W = [[rng() + eps for _ in range(k)] for _ in range(m)]
    H = [[rng() + eps for _ in range(n)] for _ in range(k)]

    curve = []
    for _ in range(iterations):
        # H <- H * (W^T V) / (W^T W H)
        Wt = _transpose(W)
        WtV = _matmul(Wt, V)
        WtW = _matmul(Wt, W)
        WtWH = _matmul(WtW, H)
        for a in range(k):
            for j in range(n):
                H[a][j] *= WtV[a][j] / (WtWH[a][j] + eps)

        # W <- W * (V H^T) / (W H H^T)
        Ht = _transpose(H)
        VHt = _matmul(V, Ht)
        HHt = _matmul(H, Ht)
        WHHt = _matmul(W, HHt)
        for i in range(m):
            for a in range(k):
                W[i][a] *= VHt[i][a] / (WHHt[i][a] + eps)

        if track_error:
            curve.append(frobenius_error(V, W, H))

    if track_error:
        return W, H, curve
    return W, H


def kl_divergence(V, W, H, eps=1e-10):
    """Generalized KL divergence D(V || W H) = sum V log(V/WH) - V + WH."""
    WH = _matmul(W, H)
    s = 0.0
    for i in range(len(V)):
        for j in range(len(V[0])):
            v = V[i][j]
            wh = WH[i][j] + eps
            if v > 0:
                s += v * math.log(v / wh) - v + wh
            else:
                s += wh
    return s


def nmf_kl(V, k, iterations=200, seed=12345, eps=1e-10, track_error=False):
    """NMF minimizing the generalized KL divergence (the objective used in probabilistic topic
    models / PLSA). Multiplicative updates normalized by column/row sums."""
    m = len(V)
    n = len(V[0])
    rng = _lcg(seed)
    W = [[rng() + eps for _ in range(k)] for _ in range(m)]
    H = [[rng() + eps for _ in range(n)] for _ in range(k)]

    curve = []
    for _ in range(iterations):
        WH = _matmul(W, H)
        # H update: H_aj <- H_aj * sum_i W_ia V_ij/(WH)_ij / sum_i W_ia
        Wt = _transpose(W)
        col_sum_W = [sum(Wt[a]) for a in range(k)]
        ratio = [[V[i][j] / (WH[i][j] + eps) for j in range(n)] for i in range(m)]
        WtR = _matmul(Wt, ratio)
        for a in range(k):
            for j in range(n):
                H[a][j] *= WtR[a][j] / (col_sum_W[a] + eps)

        WH = _matmul(W, H)
        # W update: W_ia <- W_ia * sum_j H_aj V_ij/(WH)_ij / sum_j H_aj
        row_sum_H = [sum(H[a]) for a in range(k)]
        ratio = [[V[i][j] / (WH[i][j] + eps) for j in range(n)] for i in range(m)]
        Ht = _transpose(H)
        RHt = _matmul(ratio, Ht)
        for i in range(m):
            for a in range(k):
                W[i][a] *= RHt[i][a] / (row_sum_H[a] + eps)

        if track_error:
            curve.append(kl_divergence(V, W, H))

    if track_error:
        return W, H, curve
    return W, H


def reconstruct(W, H):
    """The approximation W H."""
    return _matmul(W, H)

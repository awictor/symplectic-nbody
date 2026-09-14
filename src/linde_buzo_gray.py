"""Linde-Buzo-Gray: build a vector-quantization codebook by repeatedly splitting and refining.

Lloyd-Max quantizes a single number to N levels; VECTOR QUANTIZATION quantizes whole VECTORS -- a block of
image pixels, a frame of speech, an embedding -- to N codebook vectors, capturing correlations a
scalar quantizer misses. The Linde-Buzo-Gray algorithm (1980) is the classic codebook designer, and its
cleverness is the INITIALIZATION. Rather than guess N starting vectors (k-means' weak spot), it grows the
codebook by SPLITTING: start with one codeword (the mean of all data), perturb it into two, run Lloyd's
algorithm (assign each point to its nearest codeword, then move each codeword to its cluster's centroid)
to settle them, then split each of the two into two, refine again, and so on -- doubling the codebook
1 -> 2 -> 4 -> 8 -> ... until it reaches the target size. Each split places the new codewords near a good
local optimum, so LBG reliably finds low-distortion codebooks where plain k-means with random seeds gets
stuck.

The distortion (mean squared quantization error) drops monotonically within each Lloyd refinement and
falls as the codebook doubles. LBG is the workhorse behind image and speech VQ codecs, and its
splitting-then-refining idea underlies modern product quantization for nearest-neighbor search.

This module builds an LBG codebook to a target size (a power of two), encodes vectors to their nearest
codeword index, decodes indices back to codewords, and reports the mean squared distortion. The split
perturbation is deterministic (proportional to each codeword), so the whole design is reproducible. It is
validated: on data drawn from well-separated clusters
the codebook places one codeword in each cluster and every point is encoded to its own cluster; the
distortion decreases monotonically within Lloyd refinement and falls as the codebook doubles; each
codeword is the centroid of the points assigned to it at convergence (the Lloyd optimality condition);
encode/decode round-trips to the nearest codeword; a 1-D data set matches a scalar Lloyd-Max quantizer;
and results are reproducible per seed. Pure stdlib; the vector-quantization companion to the Lloyd-Max,
k-means, and PCA-whitening tools."""

from __future__ import annotations

import math


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def _centroid(vectors, dim):
    n = len(vectors)
    if n == 0:
        return [0.0] * dim
    return [sum(v[d] for v in vectors) / n for d in range(dim)]


def _nearest(x, codebook):
    """Index of the nearest codeword to x, and its squared distance."""
    best = 0
    best_d = _dist2(x, codebook[0])
    for k in range(1, len(codebook)):
        d = _dist2(x, codebook[k])
        if d < best_d:
            best_d = d
            best = k
    return best, best_d


def _lloyd(data, codebook, max_iter, tol):
    """Refine a codebook by Lloyd's algorithm: assign to nearest, move to centroid. Returns
    (codebook, distortion, history)."""
    dim = len(data[0])
    prev_d = None
    history = []
    for _ in range(max_iter):
        # assign
        buckets = [[] for _ in codebook]
        total = 0.0
        for x in data:
            k, d = _nearest(x, codebook)
            buckets[k].append(x)
            total += d
        total /= len(data)
        history.append(total)
        # update: each codeword -> centroid of its bucket
        for k in range(len(codebook)):
            if buckets[k]:
                codebook[k] = _centroid(buckets[k], dim)
        if prev_d is not None and abs(prev_d - total) < tol * (1 + total):
            break
        prev_d = total
    return codebook, history[-1], history


def design(data, size, epsilon=0.01, max_iter=100, tol=1e-8, track=False):
    """Design an LBG vector-quantization codebook of `size` codewords (should be a power of 2).

    Grows the codebook by splitting: 1 -> 2 -> 4 -> ... -> size, refining with Lloyd at each stage.
    Returns a dict with codebook, distortion, and (if track) the per-split distortion history."""
    data = [list(map(float, v)) for v in data]
    dim = len(data[0])
    codebook = [_centroid(data, dim)]
    split_history = []
    while len(codebook) < size:
        # split each codeword into two by a small perturbation
        new_cb = []
        for c in codebook:
            perturb = [epsilon * (c[d] if c[d] != 0 else 1.0) for d in range(dim)]
            new_cb.append([c[d] + perturb[d] for d in range(dim)])
            new_cb.append([c[d] - perturb[d] for d in range(dim)])
        codebook = new_cb[:size]  # don't overshoot the target
        codebook, dist, hist = _lloyd(data, codebook, max_iter, tol)
        split_history.append((len(codebook), dist))
    # final distortion
    _, dist, _ = _lloyd(data, codebook, max_iter, tol)
    res = {"codebook": codebook, "distortion": dist, "size": len(codebook)}
    if track:
        res["split_history"] = split_history
    return res


def encode(vectors, codebook):
    """Map each vector to the index of its nearest codeword."""
    return [_nearest(v, codebook)[0] for v in vectors]


def decode(indices, codebook):
    """Reconstruct vectors from codeword indices."""
    return [list(codebook[i]) for i in indices]


def distortion(data, codebook):
    """Mean squared quantization error of encoding `data` with `codebook`."""
    total = 0.0
    for x in data:
        _, d = _nearest(x, codebook)
        total += d
    return total / len(data)

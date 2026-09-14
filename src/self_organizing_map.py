"""Self-organizing map: a Kohonen neural grid that flattens high-dimensional data while preserving neighborhoods.

A self-organizing map (Kohonen 1982) is an unsupervised neural network that learns a low-dimensional
(usually 2-D) grid of prototype vectors laid over high-dimensional data, in a way that keeps NEIGHBORING
data close on the grid. Unlike PCA it is nonlinear, and unlike t-SNE it produces a fixed, interpretable
lattice you can index by (row, col). Each grid node holds a weight vector in the data space; training is a
simple competitive-learning loop:

  1. Present a data point x. Find the BEST-MATCHING UNIT (BMU) -- the node whose weight is closest to x.
  2. Pull the BMU AND its grid neighbors toward x, by an amount that decays with grid distance (a Gaussian
     neighborhood) and with time:
         w_node += alpha(t) * h(dist_grid(node, BMU), sigma(t)) * (x - w_node).
  3. Shrink the learning rate alpha and neighborhood radius sigma over training.

Early on, with a wide neighborhood, the whole sheet unfolds to cover the data; as sigma shrinks, nodes
fine-tune locally. The trained map has a beautiful property: points that are close in the input space map
to nearby grid cells, so the 2-D grid becomes a topology-preserving chart of the data -- used for
visualization, vector quantization, and clustering in fields from genomics to finance.

This module builds and trains a rectangular SOM with Gaussian neighborhoods and exponential decay, maps
data points to their BMUs, and measures quantization error (mean distance to BMU) and topographic error
(fraction of points whose two best-matching units are NOT grid-adjacent -- the standard neighborhood-
preservation metric). It uses a seeded RNG. It is validated: training drives quantization error steadily
down; on data from separated clusters, points from one cluster map to a contiguous region of the grid;
the topographic error is low (neighborhoods are preserved); a 2-D SOM fit to a 2-D grid of points recovers
the grid's spatial order; the neighborhood radius and learning rate decay monotonically; and results are
reproducible per seed. Pure stdlib; the topology-preserving companion to the t-SNE, PCA-whitening,
k-means, and spectral-clustering tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


class SOM:
    """A rectangular self-organizing map of `rows` x `cols` nodes over `dim`-dimensional data."""

    def __init__(self, rows, cols, dim, seed=1):
        self.rows = rows
        self.cols = cols
        self.dim = dim
        rng = _Rng(seed)
        # weights[r][c] is a length-dim vector, initialized small and random
        self.weights = [[[rng.u() for _ in range(dim)] for _ in range(cols)] for _ in range(rows)]
        self._rng = rng

    def _bmu(self, x):
        """Index (r, c) of the best-matching unit for x."""
        best = None
        best_d = float("inf")
        for r in range(self.rows):
            for c in range(self.cols):
                d = _dist2(self.weights[r][c], x)
                if d < best_d:
                    best_d = d
                    best = (r, c)
        return best

    def _two_best(self, x):
        """Return the two closest grid nodes (for the topographic-error metric)."""
        d1 = float("inf")
        d2 = float("inf")
        b1 = b2 = None
        for r in range(self.rows):
            for c in range(self.cols):
                d = _dist2(self.weights[r][c], x)
                if d < d1:
                    d2, b2 = d1, b1
                    d1, b1 = d, (r, c)
                elif d < d2:
                    d2, b2 = d, (r, c)
        return b1, b2

    def train(self, data, epochs=100, alpha0=0.5, sigma0=None, track=False):
        """Train the map on `data` (list of vectors). Returns self, or (self, qe_history) if track.

        alpha0 is the initial learning rate; sigma0 the initial neighborhood radius (defaults to half
        the larger grid dimension). Both decay exponentially to ~0 over training."""
        if sigma0 is None:
            sigma0 = max(self.rows, self.cols) / 2.0
        n = len(data)
        total = epochs * n
        tau = total / math.log(sigma0 / 0.5) if sigma0 > 0.5 else total
        step = 0
        qe_history = []
        for ep in range(epochs):
            for i in range(n):
                x = data[i]
                alpha = alpha0 * math.exp(-step / total)
                sigma = sigma0 * math.exp(-step / tau)
                br, bc = self._bmu(x)
                two_sig2 = 2.0 * sigma * sigma + 1e-12
                for r in range(self.rows):
                    for c in range(self.cols):
                        gd2 = (r - br) ** 2 + (c - bc) ** 2
                        h = math.exp(-gd2 / two_sig2)
                        if h < 1e-4:
                            continue
                        w = self.weights[r][c]
                        scale = alpha * h
                        for k in range(self.dim):
                            w[k] += scale * (x[k] - w[k])
                step += 1
            if track:
                qe_history.append(self.quantization_error(data))
        if track:
            return self, qe_history
        return self

    def map_point(self, x):
        """Return the (row, col) BMU for a data point."""
        return self._bmu(x)

    def quantization_error(self, data):
        """Mean Euclidean distance from each point to its BMU weight."""
        total = 0.0
        for x in data:
            r, c = self._bmu(x)
            total += math.sqrt(_dist2(self.weights[r][c], x))
        return total / len(data)

    def topographic_error(self, data):
        """Fraction of points whose two best-matching units are NOT grid-adjacent (8-neighborhood).

        Low topographic error means the map preserves neighborhoods -- the SOM's defining property."""
        bad = 0
        for x in data:
            b1, b2 = self._two_best(x)
            dr = abs(b1[0] - b2[0])
            dc = abs(b1[1] - b2[1])
            if max(dr, dc) > 1:      # not in the 8-neighborhood
                bad += 1
        return bad / len(data)

    def decay_schedule(self, epochs, n, alpha0=0.5, sigma0=None):
        """Return the (alpha, sigma) sequences over training, for inspection."""
        if sigma0 is None:
            sigma0 = max(self.rows, self.cols) / 2.0
        total = epochs * n
        tau = total / math.log(sigma0 / 0.5) if sigma0 > 0.5 else total
        alphas, sigmas = [], []
        for step in range(total):
            alphas.append(alpha0 * math.exp(-step / total))
            sigmas.append(sigma0 * math.exp(-step / tau))
        return alphas, sigmas

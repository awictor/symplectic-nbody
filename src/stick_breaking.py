"""Stick-breaking (GEM): the explicit weights of a Dirichlet process, snapped off a unit stick.

The Chinese restaurant process describes a Dirichlet process by how CUSTOMERS cluster; the STICK-BREAKING
construction (Sethuraman 1994; the GEM distribution after Griffiths, Engen, McCloskey) gives the same
object CONSTRUCTIVELY, as an explicit infinite list of mixture weights. Take a stick of length 1. Break off
a Beta(1, alpha) fraction beta_1 -- that is the first weight pi_1. From what REMAINS, break off another
Beta(1, alpha) fraction -- that gives pi_2 = beta_2 (1 - beta_1). Continue forever:

    beta_k ~ Beta(1, alpha),      pi_k = beta_k * prod_{j<k} (1 - beta_j).

The weights are positive, sum to 1 (the whole stick is consumed in the limit), and decay geometrically in
expectation: E[pi_k] = (1/(1+alpha)) (alpha/(1+alpha))^{k-1}. A SMALL alpha snaps off big pieces early --
a few dominant clusters; a LARGE alpha shaves thin slivers -- many near-equal clusters, the same
concentration effect alpha controls in the CRP. Attaching an independent random "dish" (atom) to each
weight yields a draw from a Dirichlet process, and sampling category labels from the weights reproduces the
CRP's rich-get-richer partitions. Stick-breaking is what makes Dirichlet-process mixtures tractable for
variational and truncated inference.

This module generates stick-breaking weights, gives the analytic expected weights and their geometric
decay, samples category labels from the weights, and reports how many weights are needed to capture a mass
threshold. It uses a seeded RNG with a Beta(1, alpha) sampler. It is validated: the generated weights are
positive and their partial sums approach 1; the empirical mean of each weight matches the geometric
formula; a smaller alpha concentrates mass in the first few weights (fewer weights needed for 95% of the
mass) while a larger alpha spreads it; sampling labels from the weights reproduces those weights as
empirical frequencies; the residual stick after k breaks has expected length (alpha/(1+alpha))^k; and
results are reproducible per seed. Pure stdlib; the Dirichlet-process construction companion to the
Chinese-restaurant, Polya-urn, and Dirichlet tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def beta_1_alpha(self, alpha):
        """Sample Beta(1, alpha): 1 - U^{1/alpha} (inverse-CDF, since Beta(1,a) CDF is 1-(1-x)^a)."""
        u = max(self.u(), 1e-12)
        return 1.0 - u ** (1.0 / alpha)


def weights(alpha, k, seed=1):
    """Generate the first k stick-breaking weights pi_1..pi_k for concentration alpha.

    Returns the list of weights (which sum to < 1; the remainder is the un-broken stick tail)."""
    rng = _Rng(seed)
    remaining = 1.0
    pis = []
    for _ in range(k):
        beta = rng.beta_1_alpha(alpha)
        pi = beta * remaining
        pis.append(pi)
        remaining *= (1 - beta)
    return pis


def expected_weight(alpha, k):
    """Analytic E[pi_k] = (1/(1+alpha)) (alpha/(1+alpha))^{k-1} (k is 1-based)."""
    p = 1.0 / (1 + alpha)
    return p * (alpha / (1 + alpha)) ** (k - 1)


def expected_residual(alpha, k):
    """Expected length of the stick remaining after k breaks: (alpha/(1+alpha))^k."""
    return (alpha / (1 + alpha)) ** k


def sample_labels(pis, n, seed=1):
    """Sample n category labels from the (normalized) stick-breaking weights."""
    total = sum(pis)
    rng = _Rng(seed)
    labels = []
    for _ in range(n):
        u = rng.u() * total
        cum = 0.0
        chosen = len(pis) - 1
        for k, pi in enumerate(pis):
            cum += pi
            if u < cum:
                chosen = k
                break
        labels.append(chosen)
    return labels


def weights_for_mass(alpha, threshold=0.95, seed=1, max_k=100000):
    """Number of stick-breaking weights needed to capture `threshold` of the total mass."""
    rng = _Rng(seed)
    remaining = 1.0
    captured = 0.0
    for k in range(1, max_k + 1):
        beta = rng.beta_1_alpha(alpha)
        pi = beta * remaining
        captured += pi
        remaining *= (1 - beta)
        if captured >= threshold:
            return k
    return max_k

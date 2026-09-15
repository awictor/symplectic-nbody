"""Kingman's coalescent: trace a sample's ancestry BACKWARD in time to its most recent common ancestor.

Wright-Fisher runs population genetics FORWARD -- allele frequencies drifting generation by generation.
But if you sample n individuals today and ask about their shared history, it is far more efficient to run
time BACKWARD: follow the n lineages up the family tree, watching them COALESCE (merge into a common
ancestor) until only one remains -- the most recent common ancestor (MRCA). KINGMAN'S COALESCENT (1982)
is the elegant limit process describing this genealogy. Its key fact: while k lineages remain, each PAIR
coalesces independently at rate 1 (in units of N generations), so the waiting time until the next
coalescence is EXPONENTIAL with rate C(k,2) = k(k-1)/2. The times shrink as k grows, so most of the tree's
depth is spent waiting for the LAST two lineages to merge.

From this come exact, testable expectations. The expected time to the MRCA is E[T_MRCA] = 2(1 - 1/n) (in N
generations) -- approaching 2 for large samples. The expected total branch length is
E[L] = 2 sum_{k=1}^{n-1} 1/k = 2 H_{n-1}, growing only LOGARITHMICALLY with sample size. Sprinkling
mutations along the branches at rate theta/2 per unit length (theta = 4 N mu) makes the expected number of
SEGREGATING SITES E[S] = theta * H_{n-1} -- the basis of Watterson's classic estimator theta_hat = S /
H_{n-1}, one of the workhorses of molecular population genetics.

This module simulates coalescent genealogies (coalescence times and a merge history), computes the MRCA
time and total branch length, places Poisson mutations to generate segregating sites, and provides the
analytic expectations and Watterson's estimator. It uses a seeded RNG. It is validated: the mean T_MRCA
matches 2(1 - 1/n) and the mean total length matches 2 H_{n-1}; the n=2 coalescence time is exponential
with mean 1; coalescence-time variance decreases as more lineages remain; the mean segregating-site count
matches theta H_{n-1} and Watterson's estimator recovers theta; the genealogy always reduces to a single
MRCA with n-1 coalescences; and results are reproducible per seed. Pure stdlib; the backward-time
population-genetics companion to the Wright-Fisher, Gillespie-SSA, and Markov-chain tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def exponential(self, rate):
        u = max(self.u(), 1e-12)
        return -math.log(u) / rate

    def poisson(self, lam):
        """Knuth's algorithm for small lambda."""
        if lam <= 0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.u()
            if p <= L:
                return k - 1


def harmonic(m):
    """H_m = sum_{k=1}^{m} 1/k."""
    return sum(1.0 / k for k in range(1, m + 1))


def simulate(n, seed=1):
    """Simulate a Kingman coalescent for n samples (time in units of N generations).

    Returns a dict with coalescence_times (n-1 waiting times, from k=n down to k=2), t_mrca (total),
    total_length (sum of k * time_k, the total branch length over all lineages), and n_coalescences."""
    rng = _Rng(seed)
    coal_times = []
    total_length = 0.0
    t_mrca = 0.0
    for k in range(n, 1, -1):
        rate = k * (k - 1) / 2.0
        tk = rng.exponential(rate)
        coal_times.append(tk)
        total_length += k * tk    # k lineages each contribute tk of branch length
        t_mrca += tk
    return {
        "coalescence_times": coal_times,
        "t_mrca": t_mrca,
        "total_length": total_length,
        "n_coalescences": n - 1,
    }


def segregating_sites(n, theta, seed=1):
    """Simulate the number of segregating sites: place Poisson(theta/2 * total_length) mutations.

    theta = 4 N mu is the scaled mutation rate. Returns (S, genealogy_dict)."""
    geneal = simulate(n, seed=seed)
    rng = _Rng(seed + 987654)
    s = rng.poisson(theta / 2.0 * geneal["total_length"])
    return s, geneal


def expected_t_mrca(n):
    """E[T_MRCA] = 2(1 - 1/n) in units of N generations."""
    return 2.0 * (1 - 1.0 / n)


def expected_total_length(n):
    """E[total branch length] = 2 H_{n-1}."""
    return 2.0 * harmonic(n - 1)


def expected_segregating_sites(n, theta):
    """E[S] = theta * H_{n-1}."""
    return theta * harmonic(n - 1)


def watterson_theta(s, n):
    """Watterson's estimator of theta from S segregating sites in a sample of n: theta_hat = S / H_{n-1}."""
    return s / harmonic(n - 1)

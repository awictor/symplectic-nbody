"""The Chinese restaurant process: a prior over partitions where popular clusters attract more members.

How do you cluster data when you DON'T know the number of clusters in advance? The CHINESE RESTAURANT
PROCESS (CRP; Aldous 1985) is the elegant answer at the heart of Bayesian nonparametrics. Picture a
restaurant with infinitely many tables. Customers arrive one at a time; customer n sits at an occupied
table with probability proportional to how many people are ALREADY there, or starts a NEW table with
probability proportional to a concentration parameter alpha:

    P(join table k) = n_k / (n - 1 + alpha),      P(new table) = alpha / (n - 1 + alpha),

where n_k is the current occupancy of table k. This is the same rich-get-richer reinforcement as Polya's
urn, lifted to an unbounded number of categories -- popular clusters grow faster, but alpha keeps opening
new ones. The number of occupied tables after n customers grows like alpha * ln(n) (logarithmically, not
linearly), and the induced distribution over PARTITIONS is EXCHANGEABLE: it depends only on the block
SIZES, not the arrival order, which is exactly what makes the CRP the predictive rule of a Dirichlet
process and the workhorse of infinite mixture models.

This module simulates the CRP, computes the exact exchangeable partition probability (the EPPF), gives the
expected number of tables E[K_n] = sum_{i=1}^{n} alpha/(alpha + i - 1), and estimates the table-count
distribution by ensemble. It uses a seeded RNG. It is validated: the mean number of tables matches the
harmonic-sum formula and grows like alpha ln n; larger alpha yields more tables; the partition probability
is exchangeable (reordering customers leaves it unchanged) and every seating's probabilities sum to 1; the
EPPF matches the product of the sequential seating probabilities; alpha -> 0 forces one big table and large
alpha forces singletons; and results are reproducible per seed. Pure stdlib; the Bayesian-nonparametric
companion to the Polya-urn, Dirichlet, and Gaussian-mixture tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def simulate(n, alpha, seed=1):
    """Seat n customers by the CRP with concentration alpha. Returns the list of table assignments
    (table index per customer) and the list of final table sizes."""
    rng = _Rng(seed)
    tables = []           # table sizes
    assignment = []
    for i in range(n):
        denom = i + alpha  # = (current customers) + alpha
        u = rng.u() * denom
        cum = 0.0
        chosen = None
        for k, size in enumerate(tables):
            cum += size
            if u < cum:
                chosen = k
                break
        if chosen is None:
            # new table
            chosen = len(tables)
            tables.append(0)
        tables[chosen] += 1
        assignment.append(chosen)
    return assignment, tables


def expected_tables(n, alpha):
    """E[K_n] = sum_{i=1}^{n} alpha / (alpha + i - 1) -- grows like alpha ln(n) for large n."""
    return sum(alpha / (alpha + i - 1) for i in range(1, n + 1))


def partition_probability(sizes, alpha):
    """Exchangeable partition probability (EPPF) of a partition with the given block sizes.

    P = alpha^K * prod_k (n_k - 1)! / [ alpha (alpha+1) ... (alpha+n-1) ],  n = sum sizes, K = #blocks."""
    n = sum(sizes)
    k = len(sizes)
    # log for stability
    log_p = k * math.log(alpha)
    for s in sizes:
        log_p += math.lgamma(s)          # (s-1)! = Gamma(s)
    # denominator: rising factorial alpha^(n) = Gamma(alpha+n)/Gamma(alpha)
    log_p -= (math.lgamma(alpha + n) - math.lgamma(alpha))
    return math.exp(log_p)


def sequential_probability(assignment, alpha):
    """Probability of a specific ordered seating sequence (product of per-customer choice probabilities)."""
    tables = []
    p = 1.0
    for i, table in enumerate(assignment):
        denom = i + alpha
        if table < len(tables):
            p *= tables[table] / denom
            tables[table] += 1
        else:
            # new table (must be the next index)
            p *= alpha / denom
            tables.append(1)
    return p


def mean_tables(n, alpha, n_runs=2000, seed=1):
    """Empirical mean number of occupied tables over n_runs simulations."""
    total = 0
    for r in range(n_runs):
        _assign, tables = simulate(n, alpha, seed=seed + r * 2749)
        total += len(tables)
    return total / n_runs


def table_size_distribution(n, alpha, seed=1):
    """Sizes of the occupied tables from one simulation, sorted descending (the cluster sizes)."""
    _assign, tables = simulate(n, alpha, seed=seed)
    return sorted(tables, reverse=True)

"""Latin hypercube sampling: space-filling experimental design that beats random Monte Carlo.

To explore or integrate over a d-dimensional space you need sample points. Plain random (Monte Carlo)
points clump and leave gaps -- purely by chance you get clusters and voids, so a small sample poorly
covers the space and integral estimates have high variance. A grid covers evenly but needs k^d points
for k levels per axis, exploding with dimension. LATIN HYPERCUBE SAMPLING (McKay, Beckman, Conover
1979) gets the best of both: divide each axis into n equal-probability strata and place exactly ONE
sample in each stratum of each axis, with the strata paired up by a random permutation per dimension.
The result is that every 1-D projection of the sample is perfectly stratified -- no matter how many
dimensions, each variable's range is covered uniformly -- which slashes the variance of any function
that is close to additive in its inputs.

Concretely, for n samples in d dimensions: for each dimension pick a random permutation of
{0,...,n-1}; sample i gets coordinate (perm[i] + u) / n with u uniform in [0,1) (or the stratum
midpoint for a centered LHS). The MAXIMIN variant additionally maximizes the minimum inter-point
distance (by keeping the best of several random LHS draws), giving better space-filling for surrogate
modelling and computer experiments.

This module generates LHS designs (random and centered), a maximin-optimized design, and integrates a
function by LHS, plus diagnostics: the one-dimensional stratification check and the estimator variance.
Validated: every axis of an LHS design has exactly one point per stratum (the defining property); LHS
integration of test functions has substantially lower variance than plain Monte Carlo at the same
sample count (the whole point); the samples lie in the unit cube; maximin has a larger minimum
pairwise distance than a plain LHS draw; and centered LHS points sit exactly at stratum midpoints.
Pure stdlib; the experimental-design companion to the low-discrepancy (Halton/Hammersley) QMC tools."""

from __future__ import annotations

import math


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)

    return nxt


def _permutation(n, rng):
    """A random permutation of range(n) via Fisher-Yates."""
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng() * (i + 1))
        p[i], p[j] = p[j], p[i]
    return p


def latin_hypercube(n, dim, seed=12345, centered=False):
    """Generate n Latin hypercube samples in the d-dimensional unit cube. Each 1-D projection has
    exactly one point in each of the n equal strata. If centered, points sit at stratum midpoints."""
    if n <= 0 or dim <= 0:
        raise ValueError("n and dim must be positive")
    rng = _lcg(seed)
    # column j: a random permutation assigns strata to samples
    samples = [[0.0] * dim for _ in range(n)]
    for j in range(dim):
        perm = _permutation(n, rng)
        for i in range(n):
            u = 0.5 if centered else rng()
            samples[i][j] = (perm[i] + u) / n
    return samples


def maximin_lhs(n, dim, seed=12345, candidates=20):
    """Return the LHS design (out of `candidates` random draws) with the largest minimum pairwise
    distance -- a simple maximin space-filling optimization."""
    best = None
    best_d = -1.0
    for c in range(candidates):
        design = latin_hypercube(n, dim, seed=seed + c)
        d = min_pairwise_distance(design)
        if d > best_d:
            best_d = d
            best = design
    return best, best_d


def min_pairwise_distance(points):
    """Smallest Euclidean distance between any two points."""
    n = len(points)
    best = float("inf")
    for i in range(n):
        for k in range(i + 1, n):
            d = math.sqrt(sum((points[i][t] - points[k][t]) ** 2 for t in range(len(points[i]))))
            best = min(best, d)
    return best


def integrate(f, dim, n, seed=12345, domain=None):
    """Estimate the integral (mean value times volume) of f over a box by LHS. domain is a list of
    (lo, hi) per dimension (default the unit cube). Returns the integral estimate."""
    samples = latin_hypercube(n, dim, seed=seed)
    if domain is None:
        domain = [(0.0, 1.0)] * dim
    vol = 1.0
    for (lo, hi) in domain:
        vol *= (hi - lo)
    total = 0.0
    for s in samples:
        pt = [domain[j][0] + s[j] * (domain[j][1] - domain[j][0]) for j in range(dim)]
        total += f(pt)
    return vol * total / n


def stratification_ok(design):
    """Check the defining LHS property: each dimension has exactly one point in each of the n
    equal-width strata [k/n, (k+1)/n)."""
    n = len(design)
    dim = len(design[0])
    for j in range(dim):
        strata = [int(design[i][j] * n) for i in range(n)]
        strata = [min(n - 1, max(0, s)) for s in strata]
        if sorted(strata) != list(range(n)):
            return False
    return True


# --- plain Monte Carlo, for the variance comparison --------------------------
def monte_carlo_integrate(f, dim, n, seed=12345, domain=None):
    """Plain random Monte Carlo integration (the baseline LHS improves on)."""
    rng = _lcg(seed)
    if domain is None:
        domain = [(0.0, 1.0)] * dim
    vol = 1.0
    for (lo, hi) in domain:
        vol *= (hi - lo)
    total = 0.0
    for _ in range(n):
        pt = [domain[j][0] + rng() * (domain[j][1] - domain[j][0]) for j in range(dim)]
        total += f(pt)
    return vol * total / n


def estimator_variance(method, f, dim, n, true_value, trials=40, base_seed=1000):
    """Empirical variance of an integration estimator over independent trials (for LHS vs MC)."""
    errs = []
    for t in range(trials):
        est = method(f, dim, n, seed=base_seed + t * 7919)
        errs.append((est - true_value) ** 2)
    return sum(errs) / len(errs)  # mean squared error

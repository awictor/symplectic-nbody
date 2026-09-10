"""Percolation: the sudden onset of connectivity.

Fill a lattice by occupying each site independently with probability p and ask: is there a
connected path of occupied sites spanning the whole system? For small p the occupied sites
form isolated islands; as p rises the islands merge, and at a sharp critical probability p_c a
single cluster suddenly spans the lattice -- the percolation threshold. It is the archetype of
a geometric phase transition: coffee brewing, forest fires spreading, oil seeping through
rock, disease jumping through a contact network, and current finding a path through a random
resistor mesh all cross the same threshold.

For 2D site percolation on a square lattice the threshold is p_c ~ 0.5927 (no closed form; a
numerical constant). Below it the largest cluster stays finite; above it a spanning cluster
appears whose fraction of occupied sites (the percolation strength) grows from zero. Near p_c
the cluster statistics are scale-free -- clusters of every size appear -- with power-law
exponents that are universal, independent of lattice details.

This module builds a random occupied lattice, finds its connected clusters by union-find,
tests whether a cluster spans top-to-bottom, measures the largest-cluster fraction, and
estimates the threshold by sweeping p, and reproduces the ~0.59 square-lattice threshold and
the sharp spanning onset. Pure stdlib (a small seeded LCG so runs are reproducible without
Math.random); the connectivity companion to the Ising and network notes.
"""

from __future__ import annotations


class _Rng:
    """Tiny reproducible LCG (numerical-recipes constants) so results don't depend on the
    environment's RNG (Math.random is unavailable in some sandboxes)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0


def occupy_lattice(n: int, p: float, seed: int = 1):
    """Return an n x n grid (list of lists of bool) with each site occupied independently with
    probability p, using a seeded reproducible RNG."""
    rng = _Rng(seed)
    return [[rng.random() < p for _ in range(n)] for _ in range(n)]


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def label_clusters(grid):
    """Union-find labelling of occupied sites into connected clusters (4-neighbour). Returns
    (parent, n): parent[] is the disjoint-set forest over flattened indices, n the grid size."""
    n = len(grid)
    parent = list(range(n * n))
    for r in range(n):
        for c in range(n):
            if not grid[r][c]:
                continue
            idx = r * n + c
            if c + 1 < n and grid[r][c + 1]:
                _union(parent, idx, idx + 1)
            if r + 1 < n and grid[r + 1][c]:
                _union(parent, idx, idx + n)
    return parent, n


def spans(grid) -> bool:
    """True if an occupied cluster connects the top row to the bottom row (vertical spanning
    path) -- i.e. the lattice percolates."""
    parent, n = label_clusters(grid)
    top_roots = {_find(parent, c) for c in range(n) if grid[0][c]}
    for c in range(n):
        if grid[n - 1][c] and _find(parent, (n - 1) * n + c) in top_roots:
            return True
    return False


def largest_cluster_fraction(grid) -> float:
    """Fraction of ALL sites that belong to the largest connected cluster. Small below p_c,
    jumping toward 1 above it (the percolation strength)."""
    parent, n = label_clusters(grid)
    counts = {}
    best = 0
    for r in range(n):
        for c in range(n):
            if grid[r][c]:
                root = _find(parent, r * n + c)
                counts[root] = counts.get(root, 0) + 1
                best = max(best, counts[root])
    return best / (n * n)


def percolation_probability(n: int, p: float, trials: int = 20, seed: int = 1) -> float:
    """Fraction of `trials` random n x n lattices at occupation p that percolate (span). Rises
    sharply through the threshold; use to locate p_c."""
    hits = 0
    for t in range(trials):
        if spans(occupy_lattice(n, p, seed=seed + t)):
            hits += 1
    return hits / trials


SQUARE_SITE_THRESHOLD = 0.5927   # 2D square-lattice site percolation threshold

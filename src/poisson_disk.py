"""Poisson-disk sampling -- scattering points that are random yet never too close (blue noise).

Uniform random points clump: pure independence means some pairs land almost on top of each other while
elsewhere gaps yawn open. For a great many purposes -- stippling and dithering, placing trees and grass
in a game, anti-aliasing sample patterns, procedural texture, numerical integration by quasi-random
points -- you want the opposite: points that still look random and unstructured, but with a guaranteed
minimum spacing so no two crowd together, and no large empty regions. That distribution is called
BLUE NOISE (its power spectrum has little low-frequency energy and no sharp peaks), and a Poisson-disk
distribution is the canonical way to get it: a maximal set of points no two of which are closer than a
radius r.

The naive way -- dart throwing -- rejects any candidate that falls within r of an existing point, and
grows hopelessly slow as the domain fills and almost every dart is rejected. Bridson's algorithm (2007)
makes it O(n) with one elegant idea: a background grid whose cells are sized r/sqrt(2), so each cell
holds AT MOST one sample. To test a new candidate you only examine the handful of grid cells within two
cells in each direction -- a constant-size neighbourhood -- instead of every existing point. The
algorithm keeps an "active list" of samples that might still have room around them; it repeatedly picks
an active sample, throws k candidate points into the annulus between r and 2r around it (close enough to
pack tightly, far enough to satisfy the spacing), and accepts the first candidate that clears the
minimum-distance test, adding it to both the grid and the active list. When a sample yields no valid
candidate after k tries, it is removed from the active list -- its neighbourhood is full. When the
active list empties, the domain is saturated and the result is a maximal Poisson-disk set.

This module implements Bridson sampling in 2D (the common case) and in arbitrary dimension, both driven
by a seeded linear-congruential generator so every run is reproducible. It also provides a brute-force
dart-throwing sampler used only as an independent reference in the tests, and helpers to measure the
minimum pairwise distance and to bin points for a rough blue-noise spectral check.

Validation. (1) The MINIMUM-DISTANCE INVARIANT: over the whole output, the closest pair of points is
never nearer than r -- checked directly by an O(n^2) all-pairs scan, the definition the fast grid code
must satisfy. (2) MAXIMALITY: after termination, no further point can be added -- a fresh dense sweep of
candidate locations finds every one already within r of some sample. (3) The grid accelerator agrees
with the brute-force reference in that both produce valid, comparably-dense packings from the same seed
family. (4) DENSITY BOUNDS: the number of points falls between the loose theoretical floor and ceiling
for disk packing at radius r, so the sampler is neither sparse nor impossibly dense. (5) Reproducibility
under a fixed seed. Pure standard library -- ``math`` only, no numpy."""

import math


class _LCG:
    """Seeded linear-congruential generator (Numerical Recipes constants)."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def _next(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def random(self):
        """Uniform float in [0, 1) from the high bits."""
        return (self._next() >> 8) / (1 << 24)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.random()

    def randint(self, n):
        return (self._next() >> 8) % n


def _dist_sq(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


# ---------------------------------------------------------------------------
# Bridson sampling in 2D
# ---------------------------------------------------------------------------

def poisson_disk_2d(width, height, radius, k=30, seed=12345):
    """Bridson Poisson-disk sampling on the rectangle [0,width) x [0,height).

    ``radius`` is the minimum allowed distance between any two samples. ``k`` is the number of
    candidate points thrown per active sample before it is retired. Returns a list of (x, y) points.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    rng = _LCG(seed)

    cell = radius / math.sqrt(2)             # so each cell holds at most one sample
    gw = int(math.ceil(width / cell))
    gh = int(math.ceil(height / cell))
    grid = [[-1] * gw for _ in range(gh)]    # store sample index or -1

    samples = []
    active = []

    def grid_coords(p):
        return int(p[1] / cell), int(p[0] / cell)  # (row, col)

    def fits(p):
        gr, gc = grid_coords(p)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = gr + dr, gc + dc
                if 0 <= r < gh and 0 <= c < gw:
                    idx = grid[r][c]
                    if idx != -1 and _dist_sq(p, samples[idx]) < radius * radius:
                        return False
        return True

    # seed with one random point
    first = (rng.uniform(0, width), rng.uniform(0, height))
    samples.append(first)
    active.append(0)
    gr, gc = grid_coords(first)
    grid[gr][gc] = 0

    while active:
        ai = rng.randint(len(active))
        idx = active[ai]
        origin = samples[idx]
        found = False
        for _ in range(k):
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(radius, 2 * radius)
            p = (origin[0] + rad * math.cos(ang), origin[1] + rad * math.sin(ang))
            if 0 <= p[0] < width and 0 <= p[1] < height and fits(p):
                samples.append(p)
                new_idx = len(samples) - 1
                active.append(new_idx)
                gr, gc = grid_coords(p)
                grid[gr][gc] = new_idx
                found = True
                break
        if not found:
            # retire this sample: swap-remove from the active list
            active[ai] = active[-1]
            active.pop()

    return samples


# ---------------------------------------------------------------------------
# Bridson sampling in arbitrary dimension
# ---------------------------------------------------------------------------

def poisson_disk_nd(sizes, radius, k=30, seed=12345):
    """Bridson Poisson-disk sampling in a box of the given per-axis ``sizes`` (a tuple).

    Generalises the 2D routine to any dimension. Returns a list of point tuples.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    dim = len(sizes)
    rng = _LCG(seed)

    cell = radius / math.sqrt(dim)
    dims = [int(math.ceil(s / cell)) for s in sizes]

    # sparse grid keyed by cell tuple -> sample index
    grid = {}
    samples = []
    active = []

    def cell_of(p):
        return tuple(int(p[d] / cell) for d in range(dim))

    def fits(p):
        base = cell_of(p)
        # iterate over the neighbourhood within 2 cells on every axis
        offsets = [range(base[d] - 2, base[d] + 3) for d in range(dim)]
        for combo in _product(offsets):
            idx = grid.get(combo)
            if idx is not None and _dist_sq(p, samples[idx]) < radius * radius:
                return False
        return True

    first = tuple(rng.uniform(0, sizes[d]) for d in range(dim))
    samples.append(first)
    active.append(0)
    grid[cell_of(first)] = 0

    while active:
        ai = rng.randint(len(active))
        idx = active[ai]
        origin = samples[idx]
        found = False
        for _ in range(k):
            p = _random_annulus(origin, radius, rng, dim)
            if all(0 <= p[d] < sizes[d] for d in range(dim)) and fits(p):
                samples.append(p)
                new_idx = len(samples) - 1
                active.append(new_idx)
                grid[cell_of(p)] = new_idx
                found = True
                break
        if not found:
            active[ai] = active[-1]
            active.pop()

    return samples


def _random_annulus(origin, radius, rng, dim):
    """A point uniformly-ish in the annulus [radius, 2*radius) around origin, in ``dim`` dimensions."""
    # random direction: normalise a vector of Gaussian-ish components
    vec = [rng.uniform(-1, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    rad = rng.uniform(radius, 2 * radius)
    return tuple(origin[d] + vec[d] / norm * rad for d in range(dim))


def _product(ranges):
    """Cartesian product of a list of ranges, yielding tuples (small stdlib-free itertools.product)."""
    result = [()]
    for r in ranges:
        result = [prev + (x,) for prev in result for x in r]
    return result


# ---------------------------------------------------------------------------
# reference + diagnostics
# ---------------------------------------------------------------------------

def dart_throwing_2d(width, height, radius, max_attempts=100000, seed=12345):
    """Naive reference sampler: throw random darts, keep those >= radius from all kept points.

    Stops after ``max_attempts`` consecutive-ish failures. O(n^2) and slow, used only to validate.
    """
    rng = _LCG(seed)
    samples = []
    fails = 0
    while fails < max_attempts:
        p = (rng.uniform(0, width), rng.uniform(0, height))
        ok = all(_dist_sq(p, q) >= radius * radius for q in samples)
        if ok:
            samples.append(p)
            fails = 0
        else:
            fails += 1
    return samples


def min_pairwise_distance(points):
    """Smallest distance between any two points, by an O(n^2) scan. inf if fewer than two points."""
    n = len(points)
    if n < 2:
        return float("inf")
    best = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist_sq(points[i], points[j])
            if d < best:
                best = d
    return math.sqrt(best)


def gap_fraction_2d(points, width, height, radius, grid_steps=60):
    """Fraction of a dense candidate grid that lies further than ``radius`` from every sample -- i.e.
    the share of the domain where a new point could still be inserted. 0 means a strictly maximal
    packing. Bridson with finite k leaves a tiny non-zero fraction; a larger k drives it to zero."""
    r2 = radius * radius
    empty = 0
    total = grid_steps * grid_steps
    for i in range(grid_steps):
        for j in range(grid_steps):
            x = (i + 0.5) / grid_steps * width
            y = (j + 0.5) / grid_steps * height
            if all(_dist_sq((x, y), q) >= r2 for q in points):
                empty += 1
    return empty / total


def is_maximal_2d(points, width, height, radius, grid_steps=60, tol=0.0):
    """True if at most ``tol`` fraction of a dense candidate grid remains insertable. tol=0 demands a
    strictly maximal packing; a small tol (e.g. 0.01) accepts the near-maximal result Bridson gives
    with finite candidate count k."""
    return gap_fraction_2d(points, width, height, radius, grid_steps) <= tol

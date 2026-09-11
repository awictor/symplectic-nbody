"""Diffusion-limited aggregation: fractal growth from random walkers.

Release a particle far from a seed and let it random-walk (Brownian motion) until it touches
the growing cluster, where it sticks forever. Repeat. What grows is not a blob but a sparse,
feathery, self-similar fractal -- because a wanderer almost always brushes an outer tip long
before it can diffuse into a fjord, the tips screen the interior and grow faster still. This
is diffusion-limited aggregation (Witten & Sander, 1981), and the same instability draws
mineral dendrites in rock, electrodeposits on an electrode, viscous fingers in a Hele-Shaw
cell, dielectric breakdown (lightning), soot, and coral.

The cluster is a fractal: its mass inside radius r grows as

    N(r) ~ r^D,   with D ~ 1.71 in two dimensions

(not 2, as a solid disc would give) -- the branches leave most of the plane empty. This
module grows an on-lattice DLA cluster (walkers released on a launch circle, killed if they
stray too far, stuck when they step next to an occupied cell), then measures the fractal
dimension by the mass-radius scaling and the radius of gyration. Pure stdlib (seeded LCG for
the walk); the fractal-growth companion to the percolation and reaction-diffusion notes.
"""

from __future__ import annotations

import math


class _Rng:
    """Seeded LCG; high bits (low bits of an LCG are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def randint(self, k: int) -> int:
        """Uniform int in [0, k)."""
        return (self._next() >> 16) % k

    def uniform(self, a: float, b: float) -> float:
        return a + (self._next() >> 8) / (1 << 24) * (b - a)


_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def grow(n_particles: int, seed: int = 1, max_radius_factor: float = 3.0):
    """Grow an on-lattice DLA cluster of up to `n_particles` cells around a seed at the origin.

    Walkers are launched from a circle just outside the current cluster, take unit random
    steps, stick when a step would land next to an occupied cell, and are discarded (a fresh
    walker launched) if they wander past `max_radius_factor` times the launch radius. Returns
    the set of occupied (x, y) integer cells including the seed.
    """
    rng = _Rng(seed)
    cluster = {(0, 0)}
    r_max = 1.0  # current cluster radius (max distance of any cell from origin)

    while len(cluster) < n_particles:
        launch_r = r_max + 5.0
        kill_r = launch_r * max_radius_factor
        # launch on a circle of radius launch_r
        ang = rng.uniform(0.0, 2.0 * math.pi)
        x = int(round(launch_r * math.cos(ang)))
        y = int(round(launch_r * math.sin(ang)))
        while True:
            # stuck? adjacent to any occupied cell
            stuck = False
            for dx, dy in _STEPS:
                if (x + dx, y + dy) in cluster:
                    stuck = True
                    break
            if stuck:
                cluster.add((x, y))
                r = math.hypot(x, y)
                if r > r_max:
                    r_max = r
                break
            # random step
            dx, dy = _STEPS[rng.randint(4)]
            x += dx
            y += dy
            if math.hypot(x, y) > kill_r:
                # wandered off; relaunch
                ang = rng.uniform(0.0, 2.0 * math.pi)
                x = int(round(launch_r * math.cos(ang)))
                y = int(round(launch_r * math.sin(ang)))
    return cluster


def center_of_mass(cluster):
    n = len(cluster)
    sx = sum(c[0] for c in cluster)
    sy = sum(c[1] for c in cluster)
    return sx / n, sy / n


def radius_of_gyration(cluster) -> float:
    """RMS distance of cells from the cluster centre of mass."""
    cx, cy = center_of_mass(cluster)
    n = len(cluster)
    return math.sqrt(sum((c[0] - cx) ** 2 + (c[1] - cy) ** 2 for c in cluster) / n)


def mass_within(cluster, r: float) -> int:
    """Number of cells within radius r of the cluster centre of mass."""
    cx, cy = center_of_mass(cluster)
    r2 = r * r
    return sum(1 for c in cluster if (c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= r2)


def fractal_dimension(cluster, n_bins: int = 12):
    """Estimate the fractal dimension D from the mass-radius scaling N(r) ~ r^D.

    Fits a straight line to log N(r) versus log r over radii spanning the cluster (dropping
    the innermost and outermost bins, where lattice discreteness and the finite edge bias the
    slope). Returns (D, [(r, N), ...]) with the sampled points.
    """
    cx, cy = center_of_mass(cluster)
    r_out = max(math.hypot(c[0] - cx, c[1] - cy) for c in cluster)
    total = len(cluster)
    pts = []
    for i in range(1, n_bins + 1):
        r = r_out * i / n_bins
        n = mass_within(cluster, r)
        if n > 0:
            pts.append((r, n))
    # Fit log N vs log r only over the scaling window: drop the innermost bin (lattice
    # discreteness) and every bin past 60% of the total mass (the finite edge saturates the
    # count there, flattening the slope below the true dimension).
    fit = [(r, n) for r, n in pts[1:] if n <= 0.6 * total]
    if len(fit) < 3:
        fit = pts[1:-1] if len(pts) > 4 else pts
    xs = [math.log(r) for r, _ in fit]
    ys = [math.log(n) for _, n in fit]
    m = len(xs)
    mx = sum(xs) / m
    my = sum(ys) / m
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    D = num / den if den else 0.0
    return D, pts


def bounds(cluster):
    """(xmin, ymin, xmax, ymax) integer bounding box of the cluster."""
    xs = [c[0] for c in cluster]
    ys = [c[1] for c in cluster]
    return min(xs), min(ys), max(xs), max(ys)

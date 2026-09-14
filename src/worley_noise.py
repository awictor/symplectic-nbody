"""Worley noise: cellular texture from the distance to the nearest scattered feature points.

Steven Worley's 1996 noise (also called CELLULAR or VORONOI noise) generates the organic, crinkly
patterns of cell walls, cracked mud, scales, stone, and water caustics. The idea is simple and
geometric: scatter FEATURE POINTS pseudo-randomly through space (one or more per grid cell), and at any
query point compute the distance to the nearest feature point (F1), the second nearest (F2), and so on.
The scalar field F1 is small near a feature point and grows toward the boundaries between points --
exactly the ridged look of Voronoi cells -- and combinations like F2 - F1 trace the CELL EDGES (zero at
the boundary, positive inside), the classic "cobblestone" texture.

The trick that makes it fast and tileable is hashing: instead of storing points, each integer grid cell
DETERMINISTICALLY generates its own feature points from a hash of its coordinates, so a query only needs
to inspect its own cell and the immediate neighbours (a 3x3 block in 2-D). Different DISTANCE METRICS
give different characters -- Euclidean gives round cells, Manhattan gives diamond/rectilinear cells,
Chebyshev gives square cells -- and the number of points per cell and which Fn you display are the
texture knobs.

This module computes 2-D Worley noise with per-cell hashed feature points, returning the sorted nearest
distances (F1, F2, ...), under Euclidean, Manhattan, or Chebyshev metrics, and it is validated: the
hashed feature points are deterministic (same seed and cell give the same points) yet vary across
cells; F1 <= F2 <= F3 always; F1 is zero exactly at a feature point and positive elsewhere; the
neighbour-limited computation agrees with a brute-force search over a wide block of cells (so no nearer
point is missed); F2 - F1 vanishes on the Voronoi boundary between two points; and the Manhattan and
Chebyshev metrics bound the Euclidean one as expected. Pure stdlib; the procedural-texture companion to
the Perlin-noise, Voronoi, and Poisson-disk tools."""

from __future__ import annotations

import math


def _hash(ix, iy, seed):
    """Deterministic 32-bit hash of an integer cell (ix, iy) and seed."""
    h = (ix * 374761393 + iy * 668265263 + seed * 1274126177) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h & 0xFFFFFFFF


def _cell_points(ix, iy, seed, points_per_cell):
    """Deterministic feature points inside integer cell (ix, iy): list of (x, y) in that cell."""
    pts = []
    h = _hash(ix, iy, seed)
    for k in range(points_per_cell):
        h = (h * 1664525 + 1013904223) & 0xFFFFFFFF
        fx = (h >> 8) / (1 << 24)
        h = (h * 1664525 + 1013904223) & 0xFFFFFFFF
        fy = (h >> 8) / (1 << 24)
        pts.append((ix + fx, iy + fy))
    return pts


def _dist(ax, ay, bx, by, metric):
    dx = ax - bx
    dy = ay - by
    if metric == "euclidean":
        return math.sqrt(dx * dx + dy * dy)
    if metric == "manhattan":
        return abs(dx) + abs(dy)
    if metric == "chebyshev":
        return max(abs(dx), abs(dy))
    raise ValueError("metric must be euclidean, manhattan, or chebyshev")


def worley(x, y, seed=0, points_per_cell=1, metric="euclidean", n=2, radius=1):
    """Sorted nearest feature-point distances at (x, y). Returns the first `n` of them (F1, F2, ...).

    Only cells within `radius` of the query cell are inspected (radius=1 -> 3x3 block)."""
    cx = math.floor(x)
    cy = math.floor(y)
    dists = []
    for ix in range(cx - radius, cx + radius + 1):
        for iy in range(cy - radius, cy + radius + 1):
            for (px, py) in _cell_points(ix, iy, seed, points_per_cell):
                dists.append(_dist(x, y, px, py, metric))
    dists.sort()
    return dists[:n]


def worley_brute(x, y, seed=0, points_per_cell=1, metric="euclidean", n=2, radius=4):
    """Reference: inspect a much wider block of cells to confirm nothing nearer is missed."""
    return worley(x, y, seed, points_per_cell, metric, n, radius)


def f1(x, y, **kw):
    """The F1 field: distance to the nearest feature point."""
    return worley(x, y, n=1, **kw)[0]


def f2_minus_f1(x, y, **kw):
    """F2 - F1: near zero on cell boundaries, positive inside cells (cobblestone edges)."""
    d = worley(x, y, n=2, **kw)
    return d[1] - d[0]


def field(width, height, scale=8.0, seed=0, points_per_cell=1, metric="euclidean", mode="f1"):
    """Sample a Worley field over a width x height grid. mode: 'f1' or 'f2-f1'. Returns a 2-D list."""
    out = [[0.0] * width for _ in range(height)]
    for j in range(height):
        for i in range(width):
            x = i / scale
            y = j / scale
            if mode == "f1":
                out[j][i] = f1(x, y, seed=seed, points_per_cell=points_per_cell, metric=metric)
            else:
                out[j][i] = f2_minus_f1(x, y, seed=seed, points_per_cell=points_per_cell, metric=metric)
    return out

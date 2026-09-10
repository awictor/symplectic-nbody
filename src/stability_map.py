"""Three-body stability maps: escape time over a grid of initial conditions.

Place two equal "primaries" and drop a third body at rest at a grid of starting
points (x, y). Integrate each and record how long until the system "ionizes" --
one body flung far away. Plotting escape time over the grid reveals the intricate,
fractal-edged boundary between bounded (possibly periodic) and escaping motion:
a picture of chaos in the space of initial conditions.

Pure stdlib; reuses NBody + the symplectic integrator.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from nbody import NBody


def escape_time(x0: float, y0: float, dt: float = 0.01, t_max: float = 40.0,
                escape_radius: float = 8.0, m_primary: float = 1.0,
                separation: float = 1.0, softening: float = 0.05) -> float:
    """Drop a massless-ish third body at (x0,y0) at rest between two primaries
    at (+/- separation/2, 0). Return the time at which any body's distance from
    the origin exceeds escape_radius, or t_max if it stays bound."""
    d = separation / 2.0
    system = NBody(
        masses=[m_primary, m_primary, 0.5 * m_primary],
        pos=[[-d, 0.0, 0.0], [d, 0.0, 0.0], [x0, y0, 0.0]],
        vel=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        softening=softening,
    )
    steps = int(t_max / dt)
    r2_esc = escape_radius * escape_radius
    for s in range(steps):
        system.step("forest_ruth", dt)
        for b in range(3):
            p = system.pos[b]
            if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] > r2_esc:
                return (s + 1) * dt
    return t_max


def scan(nx: int = 60, ny: int = 60, extent: float = 1.5,
         dt: float = 0.01, t_max: float = 40.0, **kw) -> Tuple[List[float], List[float], List[List[float]]]:
    """Compute an nx-by-ny grid of escape times over [-extent,extent]^2.
    Returns (xs, ys, grid) where grid[j][i] is the escape time at (xs[i], ys[j])."""
    xs = [-extent + 2 * extent * i / (nx - 1) for i in range(nx)]
    ys = [-extent + 2 * extent * j / (ny - 1) for j in range(ny)]
    grid = []
    for y in ys:
        row = []
        for x in xs:
            row.append(escape_time(x, y, dt=dt, t_max=t_max, **kw))
        grid.append(row)
    return xs, ys, grid


def scan_row(j: int, xs: List[float], y: float, dt: float, t_max: float, **kw) -> List[float]:
    """Compute one row of the escape-time grid (handy for parallel workers)."""
    return [escape_time(x, y, dt=dt, t_max=t_max, **kw) for x in xs]

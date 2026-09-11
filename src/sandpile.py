"""The Abelian sandpile: self-organized criticality on a grid.

Drop grains of sand one at a time onto a grid. Wherever a pile reaches 4 grains it topples,
sending one grain to each of its 4 neighbours; grains that fall off the edge are lost. A single
grain can trigger a chain reaction -- an avalanche -- that ranges from nothing to a
system-spanning cascade. Bak, Tang and Wiesenfeld introduced this in 1987 as the founding
model of self-organized criticality: with no tuning of any parameter, the system drives itself
to a critical state where avalanche sizes follow a power law, big events being rare but
unbounded. It is a candidate explanation for the scale-free statistics of earthquakes, forest
fires, and neuronal avalanches.

The toppling rule is "Abelian": the final stable configuration and the total number of
topplings do not depend on the order in which unstable sites are relaxed. The critical slope
is the toppling threshold, here 4 (the number of neighbours). Once the pile has self-organized,
adding one grain produces an avalanche whose size distribution is heavy-tailed.

This module runs the toppling dynamics: a single relaxation of a grid to its stable state
(returning the avalanche size), dropping grains at the centre, and building the classic
"sandpile identity"-adjacent fully-relaxed pattern from a tall central stack, plus the
avalanche-size series from many drops. It reproduces the conservation of grains (minus edge
loss), the Abelian independence of relaxation order, and heavy-tailed avalanche sizes. Pure
stdlib (seeded LCG for random drops); the complexity companion to the percolation and Ising
notes.
"""

from __future__ import annotations

THRESHOLD = 4


class _Rng:
    """Seeded LCG; high bits."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def randint(self, k: int) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % k


def make_grid(n: int, fill: int = 0):
    """An n x n grid initialized to `fill` grains per site."""
    return [[fill for _ in range(n)] for _ in range(n)]


def total_grains(grid) -> int:
    """Total grains currently on the grid."""
    return sum(sum(row) for row in grid)


def relax(grid):
    """Topple every site with >= 4 grains until the whole grid is stable, in place. Each
    topple moves one grain to each of the 4 neighbours; edge grains are lost. Returns the
    avalanche size = total number of topplings (order-independent, the Abelian property)."""
    n = len(grid)
    topples = 0
    unstable = [(r, c) for r in range(n) for c in range(n) if grid[r][c] >= THRESHOLD]
    while unstable:
        r, c = unstable.pop()
        if grid[r][c] < THRESHOLD:
            continue
        # topple as many full rounds as the site allows at once
        times = grid[r][c] // THRESHOLD
        grid[r][c] -= times * THRESHOLD
        topples += times
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                grid[nr][nc] += times
                if grid[nr][nc] >= THRESHOLD:
                    unstable.append((nr, nc))
    return topples


def drop_grain(grid, r, c):
    """Add one grain at (r,c) and relax; return the avalanche size (number of topplings)."""
    grid[r][c] += 1
    return relax(grid)


def is_stable(grid) -> bool:
    """True if no site is at or above the toppling threshold."""
    return all(cell < THRESHOLD for row in grid for cell in row)


def relax_stack(n: int, height: int):
    """Place `height` grains on the centre of an n x n grid and relax to the stable pattern.
    Returns (grid, total_topples). The classic self-similar sandpile figure."""
    grid = make_grid(n)
    c = n // 2
    grid[c][c] = height
    topples = relax(grid)
    return grid, topples


def avalanche_series(n: int, drops: int, seed: int = 1):
    """Drop `drops` grains at random sites on an n x n grid, relaxing after each, and return
    the list of avalanche sizes. After a transient the pile self-organizes and the sizes are
    heavy-tailed (many zeros/small, rare large cascades)."""
    rng = _Rng(seed)
    grid = make_grid(n)
    sizes = []
    for _ in range(drops):
        sizes.append(drop_grain(grid, rng.randint(n), rng.randint(n)))
    return sizes

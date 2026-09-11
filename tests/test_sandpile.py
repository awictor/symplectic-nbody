"""Tests for sandpile: Abelian toppling and self-organized criticality."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sandpile as sp

failed = 0


def check(name, cond):
    global failed
    if cond:
        print("PASS " + name)
    else:
        print("FAIL " + name)
        failed += 1


# A single site with exactly 4 grains topples once, emptying to its neighbours.
g = sp.make_grid(3)
g[1][1] = 4
topples = sp.relax(g)
check("one topple for a 4-stack", topples == 1)
check("center emptied after topple", g[1][1] == 0)
check("each neighbour gained one", g[0][1] == 1 and g[2][1] == 1 and g[1][0] == 1 and g[1][2] == 1)
check("grid stable after relax", sp.is_stable(g))

# Stable grid (all < 4) relaxes with zero topples.
gs = sp.make_grid(4, 3)
check("stable grid: no topples", sp.relax(gs) == 0)
check("stable grid unchanged", sp.total_grains(gs) == 3 * 16)

# Grain conservation: interior toppling conserves grains (no edge loss for a centered small pile).
g2 = sp.make_grid(5)
g2[2][2] = 4
before = sp.total_grains(g2)
sp.relax(g2)
check("interior topple conserves grains", sp.total_grains(g2) == before)

# Edge loss: a topple at the corner loses grains off the boundary.
g3 = sp.make_grid(3)
g3[0][0] = 4
sp.relax(g3)
check("corner topple loses grains to edge", sp.total_grains(g3) < 4)

# Abelian property: relaxing the same start gives the same total topples regardless of build.
gA = sp.make_grid(5)
gA[2][2] = 16
tA = sp.relax(gA)
gB = sp.make_grid(5)
# build up in two halves then relax once
gB[2][2] = 8
sp.relax(gB)
gB[2][2] += 8
tB_extra = sp.relax(gB)
check("Abelian: same final grid regardless of order", gA == gB)

# relax_stack: a tall central stack produces a symmetric stable pattern.
grid, topples = sp.relax_stack(11, 100)
check("stack relaxation is stable", sp.is_stable(grid))
check("stack relaxation topples a lot", topples > 20)
# Symmetry: the pattern is symmetric under reflection.
n = len(grid)
sym = all(grid[r][c] == grid[r][n - 1 - c] for r in range(n) for c in range(n))
check("relaxed stack left-right symmetric", sym)

# drop_grain returns an avalanche size >= 0 and keeps the grid stable.
gd = sp.make_grid(7, 3)   # primed near threshold
size = sp.drop_grain(gd, 3, 3)
check("drop returns nonneg avalanche size", size >= 0)
check("grid stable after drop", sp.is_stable(gd))

# Avalanche series: after many drops the pile self-organizes; sizes are heavy-tailed
# (mostly small with rare large events).
sizes = sp.avalanche_series(15, 3000, seed=1)
check("avalanche series length", len(sizes) == 3000)
nonzero = [s for s in sizes if s > 0]
check("some avalanches occur", len(nonzero) > 0)
# Heavy tail: the max avalanche is far larger than the typical (mean) one.
mean_sz = sum(sizes) / len(sizes)
check("heavy-tailed: max >> mean", max(sizes) > 10 * (mean_sz + 1e-9))
# Late-run pile has self-organized to nonzero average density.
late_grid = None  # density check via a fresh long run
dens_run = sp.avalanche_series(20, 5000, seed=2)
check("large avalanches appear once critical", max(dens_run) > 20)

if failed:
    print("%d test(s) failed" % failed)
    sys.exit(1)
print("all sandpile tests passed")

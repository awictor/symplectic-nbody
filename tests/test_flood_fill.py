"""Tests for flood_fill: queue/stack/scanline equality, barriers, connectivity, components."""

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flood_fill import (flood_fill, region_size, connected_components, component_sizes,
                        _neighbours)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- neighbours ------------------------------------------------------------
check("4-connectivity has 4 interior neighbours", len(list(_neighbours(2, 2, 5, 5, 4))) == 4)
check("8-connectivity has 8 interior neighbours", len(list(_neighbours(2, 2, 5, 5, 8))) == 8)
check("corner has 2 orthogonal neighbours", len(list(_neighbours(0, 0, 5, 5, 4))) == 2)

# --- a bounded region with a ring barrier ----------------------------------
grid = [[0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]]

g = copy.deepcopy(grid)
n = flood_fill(g, 0, 0, 9)
check("outer region fill count is 16", n == 16)
check("fill does not cross the ring barrier", g[2][2] == 0)
check("ring itself untouched", g[1][1] == 1 and g[3][3] == 1)
check("outer cells all filled", all(g[0][i] == 9 for i in range(5)))

# --- the three methods produce identical results ---------------------------
gq = copy.deepcopy(grid)
gs = copy.deepcopy(grid)
gl = copy.deepcopy(grid)
nq = flood_fill(gq, 0, 0, 9, method="queue")
ns = flood_fill(gs, 0, 0, 9, method="stack")
nl = flood_fill(gl, 0, 0, 9, method="scanline")
check("queue and stack fills match", gq == gs and nq == ns)
check("scanline matches the queue fill", gq == gl and nq == nl)

# --- filling the enclosed centre leaves the barrier and outside intact -----
gc = copy.deepcopy(grid)
flood_fill(gc, 2, 2, 7)
check("centre fill hits only the centre", gc[2][2] == 7)
check("centre fill leaves the ring", gc[1][1] == 1)
check("centre fill leaves the outside", gc[0][0] == 0)

# --- no-op when the seed already has the target value ----------------------
check("filling with the same value is a no-op", flood_fill([[5, 5], [5, 5]], 0, 0, 5) == 0)
# out of bounds seed
check("out-of-bounds seed fills nothing", flood_fill([[0]], 5, 5, 1) == 0)

# --- 4- vs 8-connectivity on a diagonal ------------------------------------
diag = [[1, 0], [0, 1]]
check("4-conn diagonal is a single cell", region_size(diag, 0, 0, 4) == 1)
check("8-conn diagonal joins the two 1s", region_size(diag, 0, 0, 8) == 2)

# --- scanline with 8-connectivity handles diagonals ------------------------
d8 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
gd = copy.deepcopy(d8)
count8 = flood_fill(gd, 0, 0, 5, connectivity=8, method="scanline")
check("scanline 8-conn fills the diagonal chain", count8 == 3)

# --- connected components: a checkerboard ----------------------------------
cb = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
_, c4 = connected_components(cb, 4)
_, c8 = connected_components(cb, 8)
check("checkerboard has 9 components under 4-connectivity", c4 == 9)
check("checkerboard has 2 components under 8-connectivity", c8 == 2)

# --- component labeling and sizes ------------------------------------------
blobs = [[1, 1, 0, 2],
         [1, 0, 0, 2],
         [0, 0, 3, 3]]
labels, count = connected_components(blobs, 4)
check("blob grid has 4 components", count == 4)
check("component sizes are correct", sorted(component_sizes(blobs, 4)) == [2, 2, 3, 5])
# every cell is labeled, same-value adjacent cells share a label
check("all cells labeled", all(labels[y][x] >= 0 for y in range(3) for x in range(4)))
check("adjacent same-value cells share a label", labels[0][0] == labels[0][1] == labels[1][0])

# --- a uniform grid is one component ---------------------------------------
uniform = [[7] * 4 for _ in range(4)]
_, cu = connected_components(uniform, 4)
check("uniform grid is one component", cu == 1)

# --- a full fill of a uniform grid recolours everything --------------------
gu = [[0] * 5 for _ in range(5)]
check("full uniform fill count is 25", flood_fill(gu, 2, 2, 3) == 25)
check("all cells recoloured", all(gu[y][x] == 3 for y in range(5) for x in range(5)))

# --- region_size does not modify the grid ----------------------------------
gr = copy.deepcopy(grid)
region_size(gr, 0, 0, 4)
check("region_size leaves the grid unchanged", gr == grid)

# --- a maze-like reachability, with a walled-off pocket --------------------
# 0 = open, 1 = wall; the top-right 0 at (3,0) is sealed off by walls
maze = [[0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 0, 0]]
gm = copy.deepcopy(maze)
flood_fill(gm, 0, 0, 2)          # flood the reachable open space from the top-left
check("maze exit is reachable", gm[3][3] == 2)
check("walled-off pocket not reached", gm[0][3] == 0)   # (x=3,y=0) sealed behind walls

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all flood_fill tests passed")

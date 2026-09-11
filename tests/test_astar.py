"""Tests for astar.py -- A* pathfinding on a grid.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Optimality is cross-checked
against Dijkstra (A* with a zero heuristic) on random grids.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import astar as A  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- heuristics -------------------------------------------------------------
check("Manhattan distance", A.manhattan((0, 0), (3, 4)) == 7)
check("Euclidean distance", approx(A.euclidean((0, 0), (3, 4)), 5.0))
check("octile distance", approx(A.octile((0, 0), (3, 3)), 3 * math.sqrt(2)))
check("zero heuristic is 0", A.zero((0, 0), (9, 9)) == 0.0)
check("Manhattan >= Euclidean (looser on a grid)", A.manhattan((0, 0), (3, 4)) >= A.euclidean((0, 0), (3, 4)))

# --- basic pathfinding ------------------------------------------------------
rows = [
    "S........",
    ".####.##.",
    ".#....#..",
    ".#.##.#.#",
    "...#....G",
]
g = A.Grid(rows)
path, cost, expanded = A.astar(g, (0, 0), (4, 8), A.manhattan)
check("A* finds a path", len(path) > 0)
check("path starts at start and ends at goal", path[0] == (0, 0) and path[-1] == (4, 8))
check("path is contiguous (unit steps)",
      all(A.manhattan(path[i], path[i + 1]) == 1 for i in range(len(path) - 1)))
check("path avoids walls", all(g.passable(c) for c in path))
check("path length matches cost + 1 (unit steps)", len(path) == cost + 1)

# --- optimality: A* == Dijkstra --------------------------------------------
_, dcost, dexp = A.dijkstra_grid(g, (0, 0), (4, 8))
check("A* cost equals Dijkstra's optimal cost", approx(cost, dcost))
check("A* expands no more nodes than Dijkstra", len(expanded) <= len(dexp))
check("A* with zero heuristic IS Dijkstra",
      A.astar(g, (0, 0), (4, 8), A.zero)[1] == dcost)

# --- unreachable goal -------------------------------------------------------
walled = A.Grid(["S#G"])
path2, cost2, _ = A.astar(walled, (0, 0), (0, 2))
check("unreachable goal returns empty path", path2 == [])
check("unreachable goal has infinite cost", cost2 == math.inf)

# --- start equals goal ------------------------------------------------------
p3, c3, _ = A.astar(g, (0, 0), (0, 0))
check("start == goal gives a trivial path", p3 == [(0, 0)] and c3 == 0.0)

# --- diagonal movement ------------------------------------------------------
diag = A.Grid(["S...", "....", "....", "...G"], diagonal=True)
pd, cd, _ = A.astar(diag, (0, 0), (3, 3), A.octile)
check("diagonal movement uses sqrt(2) steps", approx(cd, 3 * math.sqrt(2)))
check("diagonal path is shorter than 4-directional",
      cd < A.astar(A.Grid(["S...", "....", "....", "...G"]), (0, 0), (3, 3))[1])

# --- optimality on random grids (vs Dijkstra) ------------------------------
def random_grid(seed, h, w, wall_prob):
    state = seed
    cells = []
    for r in range(h):
        row = []
        for c in range(w):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            row.append("#" if (state >> 16) % 100 < wall_prob and (r, c) not in ((0, 0), (h - 1, w - 1)) else ".")
        cells.append("".join(row))
    return cells


matches = 0
tested = 0
for seed in range(1, 40):
    cells = random_grid(seed, 12, 12, 28)
    grid = A.Grid(cells)
    start, goal = (0, 0), (11, 11)
    ac = A.astar(grid, start, goal, A.manhattan)[1]
    dc = A.dijkstra_grid(grid, start, goal)[1]
    tested += 1
    if (ac == dc) or (ac == math.inf and dc == math.inf):
        matches += 1
check(f"A* matches Dijkstra's cost on all {tested} random grids", matches == tested)

# --- A* never expands more than Dijkstra on those grids --------------------
never_more = True
for seed in range(1, 40):
    grid = A.Grid(random_grid(seed, 12, 12, 28))
    ae = len(A.astar(grid, (0, 0), (11, 11), A.manhattan)[2])
    de = len(A.dijkstra_grid(grid, (0, 0), (11, 11))[2])
    if ae > de:
        never_more = False
        break
check("A* expands <= Dijkstra's node count on every random grid", never_more)

# on an open grid with the goal straight along an edge, A*'s heuristic focuses the search
# tightly while Dijkstra floods outward in all directions -- A* expands far fewer nodes.
open_grid = A.Grid(["." * 30 for _ in range(15)])
open_grid.grid[0][0] = "S"
open_grid.grid[0][29] = "G"
ae = len(A.astar(open_grid, (0, 0), (0, 29), A.manhattan)[2])
de = len(A.dijkstra_grid(open_grid, (0, 0), (0, 29))[2])
check("A* focuses toward the goal (expands far fewer than Dijkstra on open ground)", ae < de)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall astar tests passed")

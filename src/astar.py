"""A* search: Dijkstra with a sense of direction.

Dijkstra finds shortest paths by expanding outward from the start in every direction equally --
correct, but it explores a whole disc of nodes before reaching a distant goal. A* (Hart,
Nilsson & Raphael, 1968) keeps the same guarantee of optimality but adds a heuristic h(n): an
estimate of the remaining distance from n to the goal. It orders its frontier by

    f(n) = g(n) + h(n),

where g(n) is the known cost from the start and h(n) is the guess to the goal, so it expands
nodes that look like they lead toward the goal first. If the heuristic never OVERESTIMATES the
true remaining cost (it is "admissible"), A* is guaranteed to return an optimal path; if it is
also "consistent" (satisfies the triangle inequality), no node is ever expanded twice. With
h = 0 it degenerates exactly to Dijkstra; with a perfect heuristic it walks straight to the
goal.

On a grid the Manhattan distance is the admissible heuristic for 4-directional movement and the
octile/Chebyshev distance for 8-directional -- and the closer the heuristic is to the truth, the
fewer nodes A* touches. That focusing is why A* is the standard for game and robot navigation.

This module runs A* on a weighted grid with obstacles, returns the optimal path and the set of
expanded nodes, and provides the standard grid heuristics; it verifies that A* finds the same
optimal cost as Dijkstra while expanding no more nodes. Pure stdlib (a from-scratch binary
heap); the informed-search companion to the Dijkstra note.
"""

from __future__ import annotations

import math


class _MinHeap:
    """Binary min-heap of (priority, tiebreak, item); from scratch, no heapq."""

    def __init__(self):
        self._h = []

    def __len__(self):
        return len(self._h)

    def push(self, priority, tie, item):
        self._h.append((priority, tie, item))
        i = len(self._h) - 1
        h = self._h
        while i > 0:
            p = (i - 1) // 2
            if h[i][:2] < h[p][:2]:
                h[i], h[p] = h[p], h[i]
                i = p
            else:
                break

    def pop(self):
        h = self._h
        top = h[0]
        last = h.pop()
        if h:
            h[0] = last
            i, n = 0, len(h)
            while True:
                l, r, s = 2 * i + 1, 2 * i + 2, i
                if l < n and h[l][:2] < h[s][:2]:
                    s = l
                if r < n and h[r][:2] < h[s][:2]:
                    s = r
                if s == i:
                    break
                h[i], h[s] = h[s], h[i]
                i = s
        return top


# --- grid heuristics --------------------------------------------------------

def manhattan(a, b):
    """|dx| + |dy| -- admissible for 4-directional grid movement (unit steps)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    """Straight-line distance -- admissible for any movement, but loose on a 4-grid."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def octile(a, b):
    """Octile distance -- admissible for 8-directional movement with diagonal cost sqrt(2)."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def zero(a, b):
    """The trivial heuristic h=0, which turns A* into Dijkstra."""
    return 0.0


class Grid:
    """A rectangular grid of cells; '#' cells are walls. Movement is 4- or 8-directional with
    unit orthogonal cost and sqrt(2) diagonal cost."""

    def __init__(self, rows, diagonal=False):
        self.grid = [list(r) for r in rows]
        self.h = len(self.grid)
        self.w = len(self.grid[0]) if self.grid else 0
        self.diagonal = diagonal

    def passable(self, cell):
        r, c = cell
        return 0 <= r < self.h and 0 <= c < self.w and self.grid[r][c] != "#"

    def neighbors(self, cell):
        r, c = cell
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        if self.diagonal:
            d = math.sqrt(2)
            steps += [(-1, -1, d), (-1, 1, d), (1, -1, d), (1, 1, d)]
        for dr, dc, cost in steps:
            nb = (r + dr, c + dc)
            if self.passable(nb):
                yield nb, cost


def astar(grid: Grid, start, goal, heuristic=manhattan):
    """A* on the grid. Returns (path, cost, expanded) where path is the list of cells (empty if
    unreachable), cost is the total path cost (inf if unreachable), and expanded is the set of
    cells popped from the frontier (a measure of work done)."""
    open_heap = _MinHeap()
    tie = 0
    open_heap.push(heuristic(start, goal), tie, start)
    g = {start: 0.0}
    came = {start: None}
    expanded = set()
    while len(open_heap):
        _, _, current = open_heap.pop()
        if current in expanded:
            continue  # stale duplicate
        expanded.add(current)
        if current == goal:
            return _reconstruct(came, goal), g[goal], expanded
        for nb, step in grid.neighbors(current):
            ng = g[current] + step
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                came[nb] = current
                tie += 1
                open_heap.push(ng + heuristic(nb, goal), tie, nb)
    return [], math.inf, expanded


def _reconstruct(came, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came[node]
    path.reverse()
    return path


def dijkstra_grid(grid: Grid, start, goal):
    """Shortest path on the grid by plain Dijkstra (A* with the zero heuristic) -- used to
    verify A*'s optimality. Returns (path, cost, expanded)."""
    return astar(grid, start, goal, heuristic=zero)

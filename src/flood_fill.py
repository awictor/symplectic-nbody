"""Flood fill: the paint-bucket, region growing, and connected-component labeling.

Flood fill answers "which cells are reachable from here through a region of the same value?" -- the
paint-bucket tool, the region-grow of image segmentation, and the reachability query behind maze
solving and Go/Minesweeper board analysis. Starting from a seed cell, it spreads to same-valued
neighbours until it hits a boundary of different values or the grid edge.

Three strategies, all here:

  QUEUE/STACK fill: the textbook BFS (queue) or DFS (stack) that visits each reachable cell once,
      pushing its unvisited same-valued neighbours. Simple and O(cells filled).
  SCANLINE fill: the efficient variant painters use -- fill a whole horizontal RUN at once, then
      seed only the runs above and below it. Far fewer stack operations than per-cell pushing, the
      classic optimization for large flat regions.
  CONNECTED COMPONENTS: run fill from every unlabeled cell to partition the grid into maximal
      same-valued regions (the labeling step of image analysis and blob counting).

CONNECTIVITY is a parameter: 4-connected (orthogonal neighbours only) or 8-connected (including
diagonals), which changes what counts as one region -- a diagonal chain is one blob under 8-conn,
several under 4-conn.

This module implements queue, stack, and scanline flood fill (4- and 8-connected), plus
connected-component labeling and region sizing -- verified that all three fill strategies produce
identical results, that fills respect barriers and the grid edge, that 8-connectivity merges
diagonal regions that 4-connectivity separates, that filling a bounded region leaves the rest
untouched, and that component labeling counts blobs correctly (including the classic checkerboard).
Pure stdlib; a raster companion to the Bresenham note."""

from __future__ import annotations

from collections import deque


def _neighbours(x, y, w, h, connectivity):
    """In-bounds neighbours of (x, y) for 4- or 8-connectivity."""
    if connectivity == 8:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    else:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny


def flood_fill(grid, x, y, new_value, connectivity=4, method="queue"):
    """Fill the region of `grid` connected to (x, y) with `new_value`.

    grid: list of rows (grid[y][x]); modified in place and also returned. `connectivity` is 4 or 8;
    `method` is 'queue' (BFS), 'stack' (DFS), or 'scanline'. Returns the number of cells filled."""
    h = len(grid)
    if h == 0:
        return 0
    w = len(grid[0])
    if not (0 <= x < w and 0 <= y < h):
        return 0
    target = grid[y][x]
    if target == new_value:
        return 0                                        # nothing to do; avoids infinite loop
    if method == "scanline":
        return _scanline_fill(grid, x, y, target, new_value, w, h, connectivity)

    frontier = deque([(x, y)])
    grid[y][x] = new_value
    filled = 1
    while frontier:
        cx, cy = frontier.popleft() if method == "queue" else frontier.pop()
        for nx, ny in _neighbours(cx, cy, w, h, connectivity):
            if grid[ny][nx] == target:
                grid[ny][nx] = new_value
                filled += 1
                frontier.append((nx, ny))
    return filled


def _scanline_fill(grid, x, y, target, new_value, w, h, connectivity):
    """Scanline flood fill: paint whole horizontal runs, seed the rows above and below."""
    filled = 0
    stack = [(x, y)]
    while stack:
        sx, sy = stack.pop()
        if grid[sy][sx] != target:
            continue
        # extend the run left and right along this row
        left = sx
        while left > 0 and grid[sy][left - 1] == target:
            left -= 1
        right = sx
        while right < w - 1 and grid[sy][right + 1] == target:
            right += 1
        for cx in range(left, right + 1):
            grid[sy][cx] = new_value
            filled += 1
        # seed the rows above and below over the run's span (and its diagonals for 8-conn)
        span_lo = left - 1 if connectivity == 8 else left
        span_hi = right + 1 if connectivity == 8 else right
        for ny in (sy - 1, sy + 1):
            if 0 <= ny < h:
                for cx in range(max(0, span_lo), min(w, span_hi + 1)):
                    if grid[ny][cx] == target:
                        stack.append((cx, ny))
    return filled


def region_size(grid, x, y, connectivity=4):
    """Count (without modifying the grid) the cells in the region connected to (x, y)."""
    h = len(grid)
    if h == 0:
        return 0
    w = len(grid[0])
    if not (0 <= x < w and 0 <= y < h):
        return 0
    target = grid[y][x]
    seen = {(x, y)}
    frontier = deque([(x, y)])
    while frontier:
        cx, cy = frontier.popleft()
        for nx, ny in _neighbours(cx, cy, w, h, connectivity):
            if (nx, ny) not in seen and grid[ny][nx] == target:
                seen.add((nx, ny))
                frontier.append((nx, ny))
    return len(seen)


def connected_components(grid, connectivity=4):
    """Label maximal same-valued regions. Returns (labels, count): labels[y][x] is the component
    id (0-based) of each cell; count is the number of components."""
    h = len(grid)
    if h == 0:
        return [], 0
    w = len(grid[0])
    labels = [[-1] * w for _ in range(h)]
    count = 0
    for sy in range(h):
        for sx in range(w):
            if labels[sy][sx] != -1:
                continue
            target = grid[sy][sx]
            frontier = deque([(sx, sy)])
            labels[sy][sx] = count
            while frontier:
                cx, cy = frontier.popleft()
                for nx, ny in _neighbours(cx, cy, w, h, connectivity):
                    if labels[ny][nx] == -1 and grid[ny][nx] == target:
                        labels[ny][nx] = count
                        frontier.append((nx, ny))
            count += 1
    return labels, count


def component_sizes(grid, connectivity=4):
    """The size of each connected component, indexed by label."""
    labels, count = connected_components(grid, connectivity)
    sizes = [0] * count
    for row in labels:
        for lab in row:
            sizes[lab] += 1
    return sizes

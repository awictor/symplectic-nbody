"""Demo: Dijkstra's algorithm -- shortest paths on a graph and a grid.

Runs Dijkstra on a small weighted graph (checking against Bellman-Ford), then solves a grid maze
with obstacles, drawing the flood of settled distances and the shortest route found.

    python examples/dijkstra_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dijkstra import Graph, dijkstra, shortest_path, bellman_ford  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    g = Graph(undirected=True)
    for u, v, w in [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5), (3, 4, 3), (1, 4, 6)]:
        g.add_edge(u, v, w)
    dist, _ = dijkstra(g, 0)
    bf = bellman_ford(g, 0)
    print("Dijkstra: cheapest route from a source to every node\n")
    print(f"  {'node':>5}{'distance':>10}{'Bellman-Ford':>14}")
    for u in sorted(dist):
        print(f"  {u:>5}{dist[u]:>10.0f}{bf[u]:>14.0f}")
    path, cost = shortest_path(g, 0, 4)
    print(f"\n  shortest 0 -> 4: {path}, cost {cost:.0f}  (Dijkstra == Bellman-Ford: "
          f"{all(dist[k] == bf[k] for k in dist)})\n")

    # grid maze
    grid = [
        "S....#....",
        ".###.#.##.",
        ".#...#.#..",
        ".#.###.#.#",
        ".#.....#.#",
        ".#####.#.#",
        ".....#.#..",
        "####.#.##.",
        "...#...#..",
        ".#.####..G",
    ]
    path_cells, cost, dfield = solve_grid(grid)
    print(f"  Grid maze ({len(grid)}x{len(grid[0])}): shortest S->G path length = {cost}")
    print("  '#' walls, '*' the path Dijkstra found:")
    _print_grid(grid, path_cells)

    _svg(os.path.join(outdir, "dijkstra.svg"), grid, path_cells, dfield)
    print(f"\n  wrote {os.path.join(outdir, 'dijkstra.svg')}")


def solve_grid(grid):
    """Build a 4-connected grid graph (unit weights) and Dijkstra from S to G."""
    rows, cols = len(grid), len(grid[0])
    g = Graph(undirected=True)
    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "#":
                continue
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
            g.add_node((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#":
                    g.add_edge((r, c), (nr, nc), 1)
    dist, _ = dijkstra(g, start)
    path, cost = shortest_path(g, start, goal)
    return set(path), int(cost), dist


def _print_grid(grid, path_cells):
    for r, row in enumerate(grid):
        line = "    "
        for c, ch in enumerate(row):
            if (r, c) in path_cells and ch not in "SG":
                line += "*"
            else:
                line += ch
        print(line)


def _svg(path, grid, path_cells, dfield, w=760, h=420):
    rows, cols = len(grid), len(grid[0])
    finite = [d for d in dfield.values() if d < math.inf]
    dmax = max(finite) if finite else 1

    cell = 34
    x0, y0 = 40, 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Dijkstra on a grid: distance flood and the shortest route</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'cell shade = distance from S (blue near, red far); yellow = shortest path; gray = walls</text>',
    ]

    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * cell, y0 + r * cell
            ch = grid[r][c]
            if ch == "#":
                fill = "#30363d"
            elif (r, c) in path_cells:
                fill = "#ffd43b"
            else:
                d = dfield.get((r, c), math.inf)
                if d == math.inf:
                    fill = "#161b22"
                else:
                    t = d / dmax
                    # blue (#4dabf7) -> red (#ff6b6b) by distance
                    rr = int(0x4d + t * (0xff - 0x4d))
                    gg = int(0xab + t * (0x6b - 0xab))
                    bb = int(0xf7 + t * (0x6b - 0xf7))
                    fill = f"#{rr:02x}{gg:02x}{bb:02x}"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{fill}" '
                         f'stroke="#0d1117" stroke-width="1"/>')
            if ch in "SG":
                parts.append(f'<text x="{x + cell/2 - 1:.1f}" y="{y + cell/2 + 4:.1f}" '
                             f'fill="#0d1117" font-size="13" font-weight="bold" '
                             f'text-anchor="middle">{ch}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

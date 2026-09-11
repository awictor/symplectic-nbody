"""Demo: A* pathfinding -- Dijkstra with a sense of direction.

Solves a grid maze with A* and with plain Dijkstra, confirming they find the same optimal cost
while A* expands far fewer nodes. Draws the two searches side by side: the cells each one
explored and the shared optimal path.

    python examples/astar_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from astar import Grid, astar, dijkstra_grid, manhattan  # noqa: E402


def _build_maze():
    """An open field crossed by three vertical wall bands (each with a gap near the bottom), so
    the goal is reachable but the path must weave -- A*'s heuristic focuses it while Dijkstra
    floods outward."""
    rows = [["."] * 30 for _ in range(13)]
    rows[0][0] = "S"
    rows[0][29] = "G"
    for c in (8, 16, 24):
        for r in range(0, 10):   # wall from the top, leaving a gap in the lower rows
            rows[r][c] = "#"
    return ["".join(r) for r in rows]


MAZE = _build_maze()


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    grid = Grid(MAZE)
    start, goal = (0, 0), (0, 29)
    apath, acost, aexp = astar(grid, start, goal, manhattan)
    dpath, dcost, dexp = dijkstra_grid(grid, start, goal)

    print("A*: guide the search with a heuristic h(n) = estimated distance to the goal\n")
    print(f"  grid {grid.h}x{grid.w}, start {start} -> goal {goal}")
    print(f"  {'':>10}{'path cost':>11}{'nodes expanded':>16}")
    print(f"  {'A*':>10}{acost:>11.0f}{len(aexp):>16}")
    print(f"  {'Dijkstra':>10}{dcost:>11.0f}{len(dexp):>16}")
    print(f"\n  same optimal cost: {acost == dcost}   "
          f"A* expanded {100 * (1 - len(aexp) / len(dexp)):.0f}% fewer nodes")
    print("\n  A* orders its frontier by f = g + h: known cost so far plus the guess to the goal,")
    print("  so it pushes toward the goal instead of flooding outward. With an admissible")
    print("  heuristic (never overestimating) the path it returns is still guaranteed optimal --")
    print("  which is why A* is the standard for game and robot navigation.")

    _svg(os.path.join(outdir, "astar.svg"), grid, apath, aexp, dexp, start, goal)
    print(f"\n  wrote {os.path.join(outdir, 'astar.svg')}")


def _svg(path, grid, opt_path, aexp, dexp, start, goal, w=760, h=430):
    cell = 15
    gap = 40
    panel_w = grid.w * cell
    y0 = 80
    path_set = set(opt_path)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'A* vs Dijkstra: same path, far fewer nodes explored</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'blue = cells the search expanded; yellow = the optimal path; gray = walls</text>',
    ]

    def draw(x_off, expanded, title):
        parts.append(f'<text x="{x_off + panel_w / 2:.1f}" y="{y0 - 8:.1f}" fill="#e6edf3" '
                     f'font-size="12" text-anchor="middle">{title} ({len(expanded)} expanded)</text>')
        for r in range(grid.h):
            for c in range(grid.w):
                x, y = x_off + c * cell, y0 + r * cell
                ch = grid.grid[r][c]
                if ch == "#":
                    fill = "#30363d"
                elif (r, c) in path_set:
                    fill = "#ffd43b"
                elif (r, c) in expanded:
                    fill = "#1f6feb"
                else:
                    fill = "#161b22"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell-1}" height="{cell-1}" '
                             f'fill="{fill}"/>')
                if (r, c) == start or (r, c) == goal:
                    parts.append(f'<text x="{x + cell/2 - 0.5:.1f}" y="{y + cell/2 + 4:.1f}" '
                                 f'fill="#0d1117" font-size="10" font-weight="bold" '
                                 f'text-anchor="middle">{"S" if (r,c)==start else "G"}</text>')

    draw(40, aexp, "A* (Manhattan)")
    draw(40 + panel_w + gap, dexp, "Dijkstra")

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

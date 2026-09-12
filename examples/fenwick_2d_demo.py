"""Demo: 2D Fenwick tree -- dynamic rectangle sums on a grid.

Builds a grid, answers rectangle-sum queries, updates cells, and re-queries -- all in log-squared time
-- verifying against a brute grid. Draws the grid as a heatmap with a highlighted query rectangle and
its running sum.

    python examples/fenwick_2d_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fenwick_2d import Fenwick2D, from_matrix, BruteGrid  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    grid = [
        [3, 1, 4, 1, 5],
        [9, 2, 6, 5, 3],
        [5, 8, 9, 7, 9],
        [3, 2, 3, 8, 4],
    ]
    R, C = len(grid), len(grid[0])
    fw = from_matrix(grid)
    bg = BruteGrid(R, C)
    for r in range(R):
        for c in range(C):
            bg.update(r, c, grid[r][c])

    print("2D Fenwick tree: point updates and rectangle sums in O(log R * log C)\n")
    print("  grid:")
    for row in grid:
        print("    " + " ".join(f"{v:2d}" for v in row))

    print("\n  rectangle sums:")
    for (r1, c1, r2, c2) in [(0, 0, 3, 4), (1, 1, 2, 3), (0, 2, 3, 2), (2, 0, 2, 4)]:
        s = fw.rectangle_sum(r1, c1, r2, c2)
        print(f"    ({r1},{c1})-({r2},{c2}): {s:3d}  (brute {bg.rectangle_sum(r1, c1, r2, c2)})")

    print("\n  update: add 100 to cell (2,2), then re-query the center block (1,1)-(2,3):")
    before = fw.rectangle_sum(1, 1, 2, 3)
    fw.update(2, 2, 100)
    bg.update(2, 2, 100)
    after = fw.rectangle_sum(1, 1, 2, 3)
    print(f"    before {before}, after {after} (brute {bg.rectangle_sum(1, 1, 2, 3)}) "
          f"-- changed by {after - before}")

    print("\n  Both the update and the query touch only O(log R * log C) tree nodes -- so a grid whose")
    print("  cells and queries both change constantly stays fast, where a static prefix-sum table")
    print("  would cost O(R*C) to rebuild after every single update.")

    _svg(os.path.join(outdir, "fenwick_2d.svg"), grid, (1, 1, 2, 3))
    print(f"\n  wrote {os.path.join(outdir, 'fenwick_2d.svg')}")


def _svg(path, grid, rect, width=720, height=420):
    R, C = len(grid), len(grid[0])
    r1, c1, r2, c2 = rect
    cell = 60
    ox, oy = 90, 90
    vmax = max(max(row) for row in grid) or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-size="18">'
        f'2D Fenwick tree: rectangle sum of the highlighted block</text>',
        f'<text x="20" y="50" fill="#8b949e" font-size="12">'
        f'cell brightness = value; the yellow box is a query rectangle summed in log-squared time'
        f'</text>',
    ]

    for r in range(R):
        for c in range(C):
            x = ox + c * cell
            y = oy + r * cell
            v = grid[r][c]
            t = v / vmax
            # blue-to-green heat
            g = int(60 + 150 * t)
            b = int(200 - 120 * t)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" '
                         f'fill="rgb(40,{g},{b})"/>')
            parts.append(f'<text x="{x+cell/2-1:.0f}" y="{y+cell/2+4:.0f}" fill="#0d1117" '
                         f'font-size="15" text-anchor="middle" font-weight="bold">{v}</text>')

    # query rectangle outline
    rx = ox + c1 * cell
    ry = oy + r1 * cell
    rw = (c2 - c1 + 1) * cell - 2
    rh = (r2 - r1 + 1) * cell - 2
    parts.append(f'<rect x="{rx-2}" y="{ry-2}" width="{rw+4}" height="{rh+4}" fill="none" '
                 f'stroke="#ffd43b" stroke-width="3"/>')
    ssum = sum(grid[r][c] for r in range(r1, r2 + 1) for c in range(c1, c2 + 1))
    parts.append(f'<text x="{ox}" y="{oy + R*cell + 30}" fill="#ffd43b" font-size="14">'
                 f'rectangle ({r1},{c1})-({r2},{c2}) sum = {ssum}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

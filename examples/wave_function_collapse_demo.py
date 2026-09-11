"""Demo: wave function collapse generating a coherent tile map from local rules.

Generates a coastline map (land / coast / sea, where land never touches sea) and a checkerboard from
a forced rule set, confirming every adjacency obeys the rules. Draws the coastline map as a colored
tile grid.

    python examples/wave_function_collapse_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wave_function_collapse import WaveFunctionCollapse  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Wave function collapse: coherent maps from local adjacency rules\n")

    # coastline tileset: land (L) and sea (S) may never be adjacent; coast (C) separates them
    tiles = ["L", "C", "S"]
    allowed = [("L", "L"), ("L", "C"), ("C", "L"), ("C", "C"),
               ("C", "S"), ("S", "C"), ("S", "S")]
    rules = {d: set(allowed) for d in ["R", "L", "D", "U"]}
    weights = {"L": 3, "C": 1, "S": 3}

    wfc = WaveFunctionCollapse(28, 16, tiles, rules, weights=weights, seed=12)
    grid = wfc.generate()

    print(f"  Coastline map (rule: land 'L' and sea 'S' may never touch; coast 'C' between them):")
    glyph = {"L": "#", "C": ".", "S": "~"}
    for row in grid:
        print("    " + "".join(glyph[c] for c in row))
    print(f"\n  satisfies all adjacency rules: {wfc.satisfies_rules(grid)}")

    # count violations explicitly
    violations = 0
    for y in range(16):
        for x in range(28):
            for dx, dy in [(1, 0), (0, 1)]:
                nx, ny = x + dx, y + dy
                if nx < 28 and ny < 16 and {grid[y][x], grid[ny][nx]} == {"L", "S"}:
                    violations += 1
    print(f"  land-sea direct adjacencies (should be 0): {violations}")

    # checkerboard from a forced rule set
    checker = {d: {("A", "B"), ("B", "A")} for d in ["R", "L", "D", "U"]}
    cw = WaveFunctionCollapse(8, 8, ["A", "B"], checker, seed=1)
    cg = cw.generate()
    print(f"\n  Forced checkerboard rules -> a perfect 2-coloring: {cw.satisfies_rules(cg)}")

    print("\n  Each cell begins as a superposition of all tiles. WFC repeatedly collapses the")
    print("  lowest-entropy (most-constrained) cell to one tile, then propagates: neighbours lose")
    print("  any option the new choice forbids, cascading until the grid is arc-consistent. A")
    print("  contradiction (a cell with no options left) triggers a restart.")

    _svg(os.path.join(outdir, "wave_function_collapse.svg"), grid)
    print(f"\n  wrote {os.path.join(outdir, 'wave_function_collapse.svg')}")


def _svg(path, grid, width=760, height=470):
    h = len(grid)
    w = len(grid[0])
    cell = min((width - 40) / w, (height - 90) / h)
    ox = (width - w * cell) / 2
    oy = 70

    palette = {"L": "#5c8a3a", "C": "#d9c48a", "S": "#2d6fa8"}   # land green, coast sand, sea blue

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Wave function collapse: a coastline map from adjacency rules</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'green = land, sand = coast, blue = sea; land and sea never touch (coast always between)</text>',
    ]
    for y in range(h):
        for x in range(w):
            c = grid[y][x]
            px = ox + x * cell
            py = oy + y * cell
            parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cell+0.5:.1f}" '
                         f'height="{cell+0.5:.1f}" fill="{palette[c]}"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: flood fill -- paint bucket, region growing, and connected components.

Fills a bounded region on an ASCII canvas (showing the barrier is respected), confirms the queue,
stack, and scanline strategies agree, and labels the connected components of a grid -- counting
blobs and showing how 4- vs 8-connectivity changes the count.

    python examples/flood_fill_demo.py [output_dir]
"""

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flood_fill import (flood_fill, connected_components, component_sizes, region_size)  # noqa: E402


def _show(grid, mapping):
    return "\n".join("    " + "".join(mapping.get(c, str(c)) for c in row) for row in grid)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Flood fill: paint bucket, region growing, connected components\n")

    # a canvas with a wall enclosing an interior
    canvas = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    glyph = {0: ".", 1: "#", 2: "~", 3: "*"}
    print("  Canvas (# = wall, . = empty):")
    print(_show(canvas, glyph))

    # paint-bucket the interior pocket
    interior = copy.deepcopy(canvas)
    filled = flood_fill(interior, 3, 2, 2)      # seed inside the wall
    print(f"\n  Paint-bucket from inside the wall ({filled} cells filled, ~ = paint):")
    print(_show(interior, glyph))
    print("  The fill stays inside the wall -- the barrier is respected.\n")

    # the three strategies agree
    a = copy.deepcopy(canvas); na = flood_fill(a, 0, 0, 2, method="queue")
    b = copy.deepcopy(canvas); nb = flood_fill(b, 0, 0, 2, method="stack")
    c = copy.deepcopy(canvas); nc = flood_fill(c, 0, 0, 2, method="scanline")
    print(f"  Filling the OUTER region by three strategies: queue {na}, stack {nb}, "
          f"scanline {nc} cells -- all identical: {a == b == c}\n")

    # connected components of a blob grid
    blobs = [
        [1, 1, 0, 0, 2],
        [1, 0, 0, 2, 2],
        [0, 0, 3, 0, 0],
        [4, 0, 3, 3, 0],
        [4, 4, 0, 0, 5],
    ]
    labels4, count4 = connected_components(blobs, 4)
    labels8, count8 = connected_components(blobs, 8)
    print(f"  Connected components of a value grid:")
    print(f"    4-connectivity: {count4} components, sizes {sorted(component_sizes(blobs, 4))}")
    print(f"    8-connectivity: {count8} components (diagonal touches merge some)\n")

    # connectivity contrast on a diagonal chain
    diag = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    print("  A diagonal chain of 1s:")
    print(_show(diag, {0: ".", 1: "#"}))
    print(f"    region size from a corner: 4-conn = {region_size(diag, 0, 0, 4)} (isolated), "
          f"8-conn = {region_size(diag, 0, 0, 8)} (all three joined)")

    print("\n  Flood fill spreads from a seed to same-valued neighbours until it meets a boundary.")
    print("  The scanline variant paints whole horizontal runs at once, seeding only the rows above")
    print("  and below -- far fewer stack operations on big flat regions. Run it from every")
    print("  unlabeled cell and you get connected-component labeling, the atom of image analysis.")

    _svg(os.path.join(outdir, "flood_fill.svg"), canvas, interior, blobs, labels4, count4)
    print(f"\n  wrote {os.path.join(outdir, 'flood_fill.svg')}")


def _svg(path, canvas, filled, blobs, labels, ncomp, width=760, height=340):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Flood fill: paint bucket (left) and connected components (right)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: interior filled inside a wall; right: cells coloured by component label</text>',
    ]
    cell = 26

    def draw_grid(grid, ox, oy, colorfn):
        h = len(grid)
        w = len(grid[0])
        for y in range(h):
            for x in range(w):
                parts.append(f'<rect x="{ox + x * cell}" y="{oy + y * cell}" '
                             f'width="{cell - 1}" height="{cell - 1}" '
                             f'fill="{colorfn(grid[y][x], x, y)}"/>')

    def canvas_color(v, x, y):
        return {0: "#161b22", 1: "#8b949e", 2: "#4dabf7"}.get(v, "#161b22")

    draw_grid(filled, 45, 80, canvas_color)

    palette = ["#4dabf7", "#ff6b6b", "#ffd43b", "#06d6a0", "#b197fc", "#ff922b", "#8b949e",
               "#e6edf3", "#ff8fab", "#5ee0c0"]

    def comp_color(v, x, y):
        return palette[labels[y][x] % len(palette)]

    draw_grid(blobs, width // 2 + 30, 80, comp_color)
    parts.append(f'<text x="{width//2 + 30}" y="{80 + len(blobs)*cell + 18}" fill="#8b949e" '
                 f'font-size="11">{ncomp} components (4-connectivity)</text>')
    parts.append(f'<text x="45" y="{80 + len(filled)*cell + 18}" fill="#8b949e" '
                 f'font-size="11">interior filled (blue), wall grey</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

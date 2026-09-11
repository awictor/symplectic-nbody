"""Demo: Bresenham line and circle rasterization.

Rasterizes lines in several directions and a circle onto a small integer grid, printing the pixels
as ASCII art, and reports that every line stays within half a pixel of the true line and every
circle pixel within half a pixel of the true radius -- all from integer arithmetic.

    python examples/bresenham_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bresenham import line, circle, filled_circle, is_connected, max_line_error  # noqa: E402


def _ascii(pixels, w, h, mark="#"):
    grid = [[" "] * w for _ in range(h)]
    for x, y in pixels:
        if 0 <= x < w and 0 <= y < h:
            grid[h - 1 - y][x] = mark          # flip y so up is up
    return "\n".join("    |" + "".join(row) + "|" for row in grid)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Bresenham rasterization: lines and circles from integer arithmetic only\n")

    # a few lines on one grid (a fan from the origin)
    print("  Lines fanning from the bottom-left corner (integer pixels):")
    pix = set()
    for x1, y1 in [(23, 0), (23, 6), (23, 13), (15, 13), (6, 13), (0, 13)]:
        pix |= set(line(0, 0, x1, y1))
    print(_ascii(pix, 24, 14))

    print("\n  Every rasterized line is connected and within half a pixel of the true line:")
    for a, b in [((0, 0), (10, 3)), ((0, 0), (3, 10)), ((0, 0), (10, 10)), ((0, 0), (-8, 5))]:
        seg = line(a[0], a[1], b[0], b[1])
        print(f"    {a}->{b}: {len(seg):>2} pixels, connected={is_connected(seg)}, "
              f"max error {max_line_error(a[0], a[1], b[0], b[1]):.3f}")

    # a circle
    print("\n  Midpoint circle, radius 8:")
    c = circle(9, 9, 8)
    print(_ascii(c, 19, 19, "o"))
    errs = [abs(math.hypot(x - 9, y - 9) - 8) for x, y in c]
    print(f"\n    {len(c)} pixels, max deviation from the true radius: {max(errs):.3f} "
          f"(< 0.5, sub-pixel)")

    # circle scaling: pixel count tracks circumference
    print("\n  Circle pixel count tracks the circumference 2*pi*r (integer steps ~ r):")
    for r in (5, 10, 20, 40):
        n = len(circle(0, 0, r))
        print(f"    r = {r:>2}: {n:>4} pixels  (2*pi*r = {2 * math.pi * r:.0f})")

    print("\n  Bresenham tracks an integer error term: the signed distance the true line has")
    print("  drifted from the current pixel. When it crosses a threshold, step the minor axis and")
    print("  correct the error -- no floating point, no division, no rounding. The midpoint circle")
    print("  does the same with a decision variable and draws one octant, mirroring it eight ways.")

    _svg(os.path.join(outdir, "bresenham.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'bresenham.svg')}")


def _svg(path, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Bresenham: integer-pixel lines and a midpoint circle</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'each square is one pixel; lines (blue) and circle (green) rasterized with integer math '
        f'only</text>',
    ]
    px = 11                                    # pixel size in svg units
    ox, oy = 45, height - 40

    def cell(gx, gy, col):
        return (f'<rect x="{ox + gx * px:.0f}" y="{oy - (gy + 1) * px:.0f}" '
                f'width="{px - 1}" height="{px - 1}" fill="{col}"/>')

    # a fan of lines on the left
    linepix = set()
    for x1, y1 in [(28, 0), (28, 9), (28, 20), (18, 26), (6, 26), (0, 26)]:
        linepix |= set(line(0, 0, x1, y1))
    for gx, gy in linepix:
        if 0 <= gx < 30 and 0 <= gy < 28:
            parts.append(cell(gx, gy, "#4dabf7"))

    # a circle on the right
    for gx, gy in circle(44, 13, 11):
        if 30 <= gx < 62 and 0 <= gy < 28:
            parts.append(cell(gx, gy, "#06d6a0"))

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

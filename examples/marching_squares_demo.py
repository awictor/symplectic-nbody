"""Demo: marching squares extracting contour lines from a scalar field.

Contours a couple of scalar fields at several levels -- concentric circles from a radial field, and
the wavy iso-lines of a sum-of-Gaussians "terrain" -- confirming that a circle's contour has the
right radius and length. Draws the nested contour lines.

    python examples/marching_squares_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from marching_squares import contour_from_function, total_length  # noqa: E402


def _terrain(x, y):
    """A sum of Gaussian bumps and dips -- a little landscape to contour."""
    return (3.0 * math.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 4)
            + 2.0 * math.exp(-((x + 2) ** 2 + (y + 1) ** 2) / 3)
            - 1.5 * math.exp(-((x - 1) ** 2 + (y + 2) ** 2) / 2))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Marching squares: iso-contours of a scalar field\n")

    # radial field: contours are circles
    print("  Radial field f = x^2 + y^2 -- contours should be circles:")
    print(f"    {'level':>8} {'radius':>8} {'segments':>9} {'length':>9} {'2*pi*r':>9}")
    for lvl in (1.0, 4.0, 9.0, 16.0):
        segs = contour_from_function(lambda x, y: x * x + y * y, -5, 5, -5, 5, lvl, nx=120, ny=120)
        r = math.sqrt(lvl)
        print(f"    {lvl:>8.0f} {r:>8.2f} {len(segs):>9} {total_length(segs):>9.3f} "
              f"{2 * math.pi * r:>9.3f}")

    # a terrain: contours at several heights
    print("\n  Gaussian terrain -- contour-line count at several heights:")
    for h in (-0.5, 0.0, 0.5, 1.0, 2.0):
        segs = contour_from_function(_terrain, -5, 5, -5, 5, h, nx=100, ny=100)
        print(f"    height {h:>5.1f}: {len(segs):>4} contour segments, "
              f"total length {total_length(segs):.2f}")

    print("\n  Each grid cell's four corners are above or below the level -> a 4-bit case index")
    print("  selecting which edges the contour crosses; linear interpolation places the crossing")
    print("  exactly where the field equals the level, so the curve is smooth. Two ambiguous saddle")
    print("  cases are resolved by the cell-center average. This is how every contour plot is drawn.")

    circle_segs = contour_from_function(lambda x, y: x * x + y * y, -5, 5, -5, 5, 9.0,
                                        nx=120, ny=120)
    terrain_levels = [(h, contour_from_function(_terrain, -5, 5, -5, 5, h, nx=90, ny=90))
                      for h in (-0.3, 0.2, 0.7, 1.3, 2.2)]
    _svg(os.path.join(outdir, "marching_squares.svg"),
         [(1.0, contour_from_function(lambda x, y: x * x + y * y, -5, 5, -5, 5, lvl, nx=120, ny=120))
          for lvl in (1.0, 4.0, 9.0, 16.0)],
         terrain_levels)
    print(f"\n  wrote {os.path.join(outdir, 'marching_squares.svg')}")


def _svg(path, circle_levels, terrain_levels, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Marching squares: iso-contours of two scalar fields</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: circles from a radial field; right: nested iso-lines of a Gaussian terrain</text>',
    ]
    colors = ["#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b"]

    def panel(x_off, x_w, level_segs, title):
        cx = x_off + x_w / 2
        cy = height / 2 + 20
        sc = min(x_w, height - 120) / 10.4     # field spans [-5,5]

        def PX(x):
            return cx + x * sc

        def PY(y):
            return cy - y * sc

        for k, (lvl, segs) in enumerate(level_segs):
            col = colors[k % len(colors)]
            for a, b in segs:
                parts.append(f'<line x1="{PX(a[0]):.1f}" y1="{PY(a[1]):.1f}" '
                             f'x2="{PX(b[0]):.1f}" y2="{PY(b[1]):.1f}" '
                             f'stroke="{col}" stroke-width="1.4"/>')
        parts.append(f'<text x="{cx:.1f}" y="{height-24:.1f}" fill="#8b949e" font-size="11" '
                     f'text-anchor="middle">{title}</text>')

    panel(40, width / 2 - 60, circle_levels, "f = x^2 + y^2 (circles)")
    panel(width / 2, width / 2 - 40, terrain_levels, "Gaussian terrain (iso-lines)")
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: point-in-polygon testing on concave and self-intersecting shapes.

Classifies a grid of points inside/outside a concave polygon (both ray casting and winding number
agree), then shows the textbook case where they DISAGREE: the centre of a pentagram, which even-odd
ray parity calls outside but the nonzero-winding rule calls inside. Draws the polygons with points
coloured by membership.

    python examples/point_in_polygon_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from point_in_polygon import (ray_casting, winding_number, winding_count,  # noqa: E402
                              signed_area, area, centroid)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    # a concave "arrow" polygon
    arrow = [(0, 0), (6, 3), (0, 6), (2, 3)]
    print("Point-in-polygon: ray casting vs winding number\n")
    print(f"  Concave arrow polygon, area {area(arrow):.1f}, "
          f"centroid ({centroid(arrow)[0]:.2f}, {centroid(arrow)[1]:.2f}), "
          f"orientation {'CCW' if signed_area(arrow) > 0 else 'CW'}\n")

    # classify a coarse grid, count inside
    inside = 0
    total = 0
    disagreements = 0
    for gx in range(7):
        for gy in range(7):
            p = (gx + 0.5, gy + 0.5)
            r = ray_casting(p, arrow)
            w = winding_number(p, arrow)
            total += 1
            if r:
                inside += 1
            if r != w:
                disagreements += 1
    print(f"  Over a 7x7 grid: {inside}/{total} points inside; ray casting and winding")
    print(f"  number agree on all of them ({disagreements} disagreements) -- as they must for a")
    print(f"  simple polygon.\n")

    # the pentagram: the classic divergence
    pent = [(math.cos(math.pi / 2 + 2 * math.pi * k / 5),
             math.sin(math.pi / 2 + 2 * math.pi * k / 5)) for k in range(5)]
    pentagram = [pent[(2 * k) % 5] for k in range(5)]     # one-stroke 5-pointed star
    print("  Self-intersecting PENTAGRAM (a 5-point star drawn in one stroke):")
    print("  its centre pentagon is enclosed by TWO loops, so the two rules disagree --")
    print(f"    ray casting (even-odd):   centre is {'inside' if ray_casting((0, 0), pentagram) else 'OUTSIDE'}")
    print(f"    winding number (nonzero): centre is {'INSIDE' if winding_number((0, 0), pentagram) else 'outside'}")
    print(f"    winding count at centre:  {winding_count((0, 0), pentagram)} (wrapped twice)\n")

    print("  Ray casting counts edge crossings (odd = in); winding sums the signed turns of the")
    print("  polygon around the point (nonzero = in). They agree on any simple polygon; on a")
    print("  self-overlapping one, even-odd cancels a doubly-wrapped region to 'out' while winding")
    print("  keeps it 'in'. Both handle concavity that a convex side-test cannot.")

    _svg(os.path.join(outdir, "point_in_polygon.svg"), arrow, pentagram)
    print(f"\n  wrote {os.path.join(outdir, 'point_in_polygon.svg')}")


def _svg(path, arrow, pentagram, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Point-in-polygon: concave (agree) vs self-intersecting (disagree)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: grid points green=inside, grey=outside a concave arrow; right: pentagram centre '
        f'(the rules split)</text>',
    ]

    # left panel: arrow with classified grid
    lx0, lx1 = 45, width // 2 - 20
    y0, y1 = height - 40, 70
    xa, xb, ya, yb = -1, 7, -1, 7

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(y):
        return y0 - (y - ya) / (yb - ya) * (y0 - y1)

    ap = " ".join(f"{LX(v[0]):.1f},{LY(v[1]):.1f}" for v in arrow)
    parts.append(f'<polygon points="{ap}" fill="#4dabf7" fill-opacity="0.1" '
                 f'stroke="#4dabf7" stroke-width="2"/>')
    for gx in range(7):
        for gy in range(7):
            p = (gx + 0.5, gy + 0.5)
            inside = ray_casting(p, arrow)
            parts.append(f'<circle cx="{LX(p[0]):.1f}" cy="{LY(p[1]):.1f}" r="2.6" '
                         f'fill="{"#06d6a0" if inside else "#484f58"}"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">concave arrow (ray = winding)</text>')

    # right panel: pentagram
    rx0, rx1 = width // 2 + 30, width - 30
    rc = (rx0 + rx1) / 2
    rcy = (y0 + y1) / 2
    sc = min((rx1 - rx0), (y0 - y1)) / 2.6

    def PX(x):
        return rc + x * sc

    def PY(y):
        return rcy - y * sc

    pp = " ".join(f"{PX(v[0]):.1f},{PY(v[1]):.1f}" for v in pentagram)
    parts.append(f'<polygon points="{pp}" fill="#ff6b6b" fill-opacity="0.12" '
                 f'stroke="#ff6b6b" stroke-width="2" fill-rule="nonzero"/>')
    # centre marker
    parts.append(f'<circle cx="{PX(0):.1f}" cy="{PY(0):.1f}" r="5" fill="#ffd43b" '
                 f'stroke="#0d1117" stroke-width="1"/>')
    parts.append(f'<text x="{PX(0):.1f}" y="{PY(0)-10:.1f}" fill="#ffd43b" font-size="10" '
                 f'text-anchor="middle">centre: winding=in, even-odd=out</text>')
    parts.append(f'<text x="{rc:.1f}" y="{y0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">pentagram (rules disagree)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

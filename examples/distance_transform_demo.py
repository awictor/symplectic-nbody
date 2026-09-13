"""Demo: exact Euclidean distance transform (Felzenszwalb-Huttenlocher).

Takes a small binary image with a few feature pixels, computes the exact distance from every pixel to
the nearest feature in linear time, verifies it against brute force, and draws the distance field as a
heatmap with the feature pixels marked.

    python examples/distance_transform_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distance_transform import (  # noqa: E402
    distance_transform_2d,
    squared_distance_transform_2d,
    brute_squared_distance_2d,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Exact Euclidean distance transform: distance to nearest feature, in O(n)\n")

    # a 12x20 image with a few scattered feature pixels
    h, w = 12, 20
    feature = [[False] * w for _ in range(h)]
    sites = [(2, 3), (5, 15), (9, 8), (1, 18), (10, 2)]
    for (i, j) in sites:
        feature[i][j] = True

    print(f"  {h}x{w} image, {len(sites)} feature pixels at {sites}.\n")

    d = distance_transform_2d(feature)
    sq = squared_distance_transform_2d(feature)
    bsq = brute_squared_distance_2d(feature)
    matches = all(abs(sq[i][j] - bsq[i][j]) < 1e-9 for i in range(h) for j in range(w))
    print(f"  Matches brute-force nearest-feature search at every pixel: {matches}")

    # show a small slice as ASCII
    print("\n  Distance field (rounded), features shown as '#':")
    for i in range(h):
        row = ""
        for j in range(w):
            if feature[i][j]:
                row += "  #"
            else:
                row += f"{int(round(d[i][j])):>3}"
        print("   " + row)

    maxd = max(d[i][j] for i in range(h) for j in range(w))
    print(f"\n  Farthest pixel is {maxd:.2f} from any feature.")
    print("  Separable trick: 1-D lower-envelope-of-parabolas transform down columns then across")
    print("  rows gives the exact 2-D Euclidean distance in linear time, no per-pair search.")

    _svg(os.path.join(outdir, "distance_transform.svg"), feature, d)
    print(f"\n  wrote {os.path.join(outdir, 'distance_transform.svg')}")


def _svg(path, feature, d, width=760, height=340):
    h = len(d)
    w = len(d[0])
    cell = min((width - 40) // w, (height - 60) // h)
    ox, oy = 20, 40
    maxd = max(d[i][j] for i in range(h) for j in range(w)) or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Distance to nearest feature (dark = close, bright = far); features ringed</text>',
    ]
    for i in range(h):
        for j in range(w):
            t = d[i][j] / maxd
            # blue (near) -> yellow (far) ramp
            r = int(0x0d + t * (0xff - 0x0d))
            g = int(0x30 + t * (0xd4 - 0x30))
            b = int(0x60 + t * (0x3b - 0x60))
            parts.append(f'<rect x="{ox + j*cell}" y="{oy + i*cell}" width="{cell-1}" '
                         f'height="{cell-1}" fill="rgb({r},{g},{b})"/>')
            if feature[i][j]:
                cx = ox + j * cell + cell / 2
                cy = oy + i * cell + cell / 2
                parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{cell/2.5:.1f}" '
                             f'fill="none" stroke="#ff6b6b" stroke-width="1.5"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

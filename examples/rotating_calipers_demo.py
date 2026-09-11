"""Demo: rotating calipers -- diameter, width, and the minimum-area bounding box.

Generates a random point cloud, computes its convex hull, the farthest pair (diameter), the width
(thinnest slab), and the minimum-area enclosing rectangle, then draws all of them.

    python examples/rotating_calipers_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotating_calipers import convex_hull, diameter, width, min_area_rectangle  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Rotating calipers: diameter, width, and minimum-area bounding box\n")

    # a random-ish point cloud (seeded, elongated on a diagonal to make the min box interesting)
    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    pts = []
    for _ in range(40):
        t = rng()
        # elongated blob along a diagonal, with some spread
        x = 40 + t * 220 + (rng() - 0.5) * 60
        y = 60 + t * 130 + (rng() - 0.5) * 60
        pts.append((x, y))

    hull = convex_hull(pts)
    p, q, diam = diameter(pts)
    w = width(pts)
    rect = min_area_rectangle(pts)

    print(f"  {len(pts)} points, hull has {len(hull)} vertices")
    print(f"  diameter (farthest pair): {diam:.2f}")
    print(f"    between ({p[0]:.1f}, {p[1]:.1f}) and ({q[0]:.1f}, {q[1]:.1f})")
    print(f"  width (thinnest slab): {w:.2f}")
    print(f"  minimum-area bounding box:")
    print(f"    area {rect['area']:.1f}, {rect['width']:.1f} x {rect['height']:.1f}, "
          f"perimeter {rect['perimeter']:.1f}")

    # compare to the axis-aligned box
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    aabb = (max(xs) - min(xs)) * (max(ys) - min(ys))
    print(f"    axis-aligned box area {aabb:.1f} -> the rotated box saves "
          f"{100 * (1 - rect['area'] / aabb):.1f}%")

    print("\n  A single rotation of the calipers enumerates every antipodal pair (the diameter is")
    print("  always among them), the thinnest slab gives the width, and the minimum-area rectangle")
    print("  must have a side flush with a hull edge -- so all three fall out of one hull sweep.")

    _svg(os.path.join(outdir, "rotating_calipers.svg"), pts, hull, (p, q), rect)
    print(f"\n  wrote {os.path.join(outdir, 'rotating_calipers.svg')}")


def _svg(path, pts, hull, diam_pair, rect, width_px=760, height_px=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}" font-family="monospace">',
        f'<rect width="{width_px}" height="{height_px}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Rotating calipers: hull, diameter, and minimum-area bounding box</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'purple = min-area rectangle, red = diameter (farthest pair), green = convex hull</text>',
    ]

    def fy(y):
        return height_px - 30 - (y - 40) * 0.9        # flip y, small margin

    def fx(x):
        return 30 + (x - 20) * 0.9

    # minimum-area rectangle
    corners = rect["corners"]
    poly = " ".join(f"{fx(c[0]):.1f},{fy(c[1]):.1f}" for c in corners)
    parts.append(f'<polygon points="{poly}" fill="#b197fc" fill-opacity="0.10" '
                 f'stroke="#b197fc" stroke-width="1.8"/>')

    # convex hull
    hpoly = " ".join(f"{fx(h[0]):.1f},{fy(h[1]):.1f}" for h in hull)
    parts.append(f'<polygon points="{hpoly}" fill="none" stroke="#06d6a0" stroke-width="2"/>')

    # points
    for x, y in pts:
        parts.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3" fill="#4dabf7"/>')

    # diameter
    (px, py), (qx, qy) = diam_pair
    parts.append(f'<line x1="{fx(px):.1f}" y1="{fy(py):.1f}" x2="{fx(qx):.1f}" y2="{fy(qy):.1f}" '
                 f'stroke="#ff6b6b" stroke-width="2.2" stroke-dasharray="6,4"/>')
    parts.append(f'<circle cx="{fx(px):.1f}" cy="{fy(py):.1f}" r="5" fill="#ff6b6b"/>')
    parts.append(f'<circle cx="{fx(qx):.1f}" cy="{fy(qy):.1f}" r="5" fill="#ff6b6b"/>')

    # legend
    parts.append(f'<text x="20" y="{height_px-16}" fill="#8b949e" font-size="12">'
                 f'min-area box area {rect["area"]:.0f} vs axis-aligned box (rotated to fit the cloud)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

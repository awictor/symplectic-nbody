"""Demo: the convex hull of a scatter of points.

Computes the convex hull (the rubber-band boundary) of random points by Andrew's monotone chain,
reports its area, perimeter, and diameter (farthest pair), and confirms every point lies inside.
Draws the points with the hull outlined and the diameter marked.

    python examples/convex_hull_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from convex_hull import (convex_hull, polygon_area, perimeter, point_in_hull,  # noqa: E402
                         diameter, is_convex_ccw)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # a blob of points
    n = 60
    pts = [(rng() * 100, rng() * 100) for _ in range(n)]

    hull = convex_hull(pts)
    a, b, dia = diameter(pts)

    print("Convex hull: the tightest polygon enclosing a set of points\n")
    print(f"  {n} random points -> hull of {len(hull)} vertices "
          f"({100 * len(hull) / n:.0f}% of the points are on the boundary)\n")
    print(f"  hull is convex & counter-clockwise: {is_convex_ccw(hull)}")
    print(f"  every input point inside or on the hull: {all(point_in_hull(p, hull) for p in pts)}\n")
    print(f"  enclosed area:  {polygon_area(hull):.1f}")
    print(f"  perimeter:      {perimeter(hull):.1f}")
    print(f"  diameter (farthest pair): {dia:.1f}  between "
          f"({a[0]:.1f},{a[1]:.1f}) and ({b[0]:.1f},{b[1]:.1f})")

    print("\n  Hull vertices in counter-clockwise order:")
    for v in hull:
        print(f"    ({v[0]:6.1f}, {v[1]:6.1f})")

    # a few shapes to show interior-point dropping
    print("\n  Interior points are dropped -- only the extreme points survive:")
    for name, shape in [("square + centre", [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]),
                        ("collinear", [(0, 0), (1, 1), (2, 2), (3, 3)]),
                        ("triangle + inside", [(0, 0), (4, 0), (2, 4), (2, 1)])]:
        h = convex_hull(shape)
        print(f"    {name:>18}: {len(shape)} points -> {len(h)} hull vertices")

    print("\n  Andrew's monotone chain sorts the points, then sweeps building the lower and upper")
    print("  hulls, keeping only left turns (positive cross product) and popping any vertex that")
    print("  would make a right turn. O(n log n), dominated by the sort. The farthest pair always")
    print("  lies on the hull, so the diameter needs only the hull vertices.")

    _svg(os.path.join(outdir, "convex_hull.svg"), pts, hull, a, b)
    print(f"\n  wrote {os.path.join(outdir, 'convex_hull.svg')}")


def _svg(path, pts, hull, dia_a, dia_b, width=760, height=430):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xa, xb = min(xs) - 4, max(xs) + 4
    ya, yb = min(ys) - 4, max(ys) + 4
    px0, px1 = 45, width - 30
    py0, py1 = height - 40, 70

    def X(x):
        return px0 + (x - xa) / (xb - xa) * (px1 - px0)

    def Y(y):
        return py0 - (y - ya) / (yb - ya) * (py0 - py1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Convex hull: the rubber-band boundary of a point set</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'hull vertices in blue, interior points grey, the diameter (farthest pair) dashed '
        f'yellow</text>',
    ]
    # filled hull polygon
    poly = " ".join(f"{X(v[0]):.1f},{Y(v[1]):.1f}" for v in hull)
    parts.append(f'<polygon points="{poly}" fill="#4dabf7" opacity="0.12" '
                 f'stroke="#4dabf7" stroke-width="2"/>')
    # diameter line
    parts.append(f'<line x1="{X(dia_a[0]):.1f}" y1="{Y(dia_a[1]):.1f}" '
                 f'x2="{X(dia_b[0]):.1f}" y2="{Y(dia_b[1]):.1f}" stroke="#ffd43b" '
                 f'stroke-width="1.6" stroke-dasharray="5 3"/>')
    # points
    hullset = set(map(tuple, hull))
    for p in pts:
        on = tuple(p) in hullset
        parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="{3.5 if on else 2.5:.1f}" '
                     f'fill="{"#4dabf7" if on else "#8b949e"}" '
                     f'{"stroke=#0d1117 stroke-width=0.6" if on else "opacity=0.6"}/>')
    # diameter endpoints highlighted
    for p in (dia_a, dia_b):
        parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="5" fill="none" '
                     f'stroke="#ffd43b" stroke-width="1.6"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

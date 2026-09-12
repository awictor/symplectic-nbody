"""Demo: GJK collision detection -- do two convex shapes overlap, via the Minkowski difference.

Tests convex polygons for collision with GJK, verifies against the Separating Axis Theorem, and shows
that two shapes collide exactly when their Minkowski difference contains the origin. Draws a colliding
and a separated pair.

    python examples/gjk_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gjk import (intersects, sat_intersects, minkowski_contains_origin,  # noqa: E402
                 minkowski_difference)


def ngon(cx, cy, r, n, rot=0.0):
    return [(cx + r * math.cos(2 * math.pi * k / n + rot),
             cy + r * math.sin(2 * math.pi * k / n + rot)) for k in range(n)]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("GJK: convex collision detection via the Minkowski difference\n")

    cases = [
        ("overlapping squares", [(0, 0), (3, 0), (3, 3), (0, 3)], [(2, 2), (5, 2), (5, 5), (2, 5)]),
        ("touching at a corner", [(0, 0), (2, 0), (2, 2), (0, 2)], [(2, 2), (4, 2), (4, 4), (2, 4)]),
        ("far apart", [(0, 0), (2, 0), (1, 2)], [(10, 10), (12, 10), (11, 12)]),
        ("pentagon in hexagon", ngon(0, 0, 1, 5), ngon(0, 0, 4, 6)),
    ]
    print(f"  {'case':24s} {'GJK':>6} {'SAT':>6} {'Minkowski':>10}")
    for name, a, b in cases:
        g = intersects(a, b)
        s = sat_intersects(a, b)
        m = minkowski_contains_origin(a, b)
        print(f"  {name:24s} {str(g):>6} {str(s):>6} {str(m):>10}")

    print("\n  Two convex shapes overlap iff their Minkowski difference A(-)B = {a - b} contains the")
    print("  origin. GJK never builds that difference: it queries SUPPORT points (the farthest vertex")
    print("  in a direction) to grow a simplex toward the origin, deciding overlap in a few iterations")
    print("  no matter how many vertices the shapes have. It is the collision core of physics engines.")

    a = [(0, 0), (4, 0), (4, 3), (0, 3)]
    b = [(2, 1), (6, 1), (6, 4), (2, 4)]
    print(f"\n  example A x B collide: {intersects(a, b)} (their Minkowski difference straddles 0)")

    _svg(os.path.join(outdir, "gjk.svg"), a, b)
    print(f"\n  wrote {os.path.join(outdir, 'gjk.svg')}")


def _svg(path, a, b, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'GJK: two shapes collide iff their Minkowski difference contains the origin</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'left: the two overlapping shapes A, B. right: A(-)B with the origin (red) inside it</text>',
    ]

    # left panel: the two shapes
    ox1, oy1, sc1 = 60, 200, 22

    def L(x, y):
        return (ox1 + x * sc1, oy1 - y * sc1)

    parts.append(_poly(a, L, "#4dabf7", "A"))
    parts.append(_poly(b, L, "#06d6a0", "B"))

    # right panel: the Minkowski difference and origin
    md = minkowski_difference(a, b)
    from convex_hull import convex_hull
    hull = convex_hull(md)
    ox2, oy2, sc2 = 500, 200, 14

    def R(x, y):
        return (ox2 + x * sc2, oy2 - y * sc2)

    parts.append(_poly(hull, R, "#ffd43b", "A(-)B"))
    for p in md:
        x, y = R(p[0], p[1])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="#8b949e"/>')
    ox, oy = R(0, 0)
    parts.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="5" fill="#ff6b6b"/>')
    parts.append(f'<text x="{ox+8:.0f}" y="{oy-6:.0f}" fill="#ff6b6b" font-size="11">origin</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def _poly(poly, transform, color, label):
    pts = " ".join(f"{transform(x, y)[0]:.1f},{transform(x, y)[1]:.1f}" for x, y in poly)
    cx = sum(transform(x, y)[0] for x, y in poly) / len(poly)
    cy = sum(transform(x, y)[1] for x, y in poly) / len(poly)
    return (f'<polygon points="{pts}" fill="{color}" opacity="0.35" stroke="{color}" '
            f'stroke-width="2"/><text x="{cx:.0f}" y="{cy:.0f}" fill="{color}" font-size="13" '
            f'text-anchor="middle">{label}</text>')


if __name__ == "__main__":
    main()

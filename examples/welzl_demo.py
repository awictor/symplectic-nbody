"""Demo: Welzl's smallest enclosing circle.

Finds the minimum enclosing circle of a point cloud, identifies the 2-3 support points that pin it
down, confirms all points are inside, and draws the cloud with the circle and its support points.

    python examples/welzl_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from welzl import smallest_enclosing_circle, is_enclosing, _dist  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Welzl's algorithm: the smallest circle enclosing a point cloud\n")

    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    pts = [(40 + rng() * 300, 40 + rng() * 250) for _ in range(50)]
    (cx, cy), r = smallest_enclosing_circle(pts)

    print(f"  {len(pts)} points")
    print(f"  smallest enclosing circle: centre ({cx:.1f}, {cy:.1f}), radius {r:.2f}")
    print(f"  all points inside: {is_enclosing(pts, ((cx, cy), r))}")

    # find the support points (those on the boundary)
    support = [p for p in pts if abs(_dist(p, (cx, cy)) - r) < 1e-6]
    print(f"  support points (on the boundary): {len(support)} -- "
          f"the circle is pinned by {'2 (a diameter)' if len(support) == 2 else '3 (a circumcircle)'}")

    # compare to the naive bounding box's circumscribed circle
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    box_r = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 2
    print(f"\n  For comparison, the bounding-box circumscribed circle has radius {box_r:.2f}")
    print(f"    -> the true minimum circle is {100*(1-r/box_r):.0f}% smaller in radius")

    print("\n  The smallest circle is determined by at most 3 boundary points. Welzl processes points")
    print("  in random order; a point inside the current circle is skipped, but one outside must lie")
    print("  on the new circle's boundary, so the circle is rebuilt with it fixed there -- O(n) expected.")

    _svg(os.path.join(outdir, "welzl.svg"), pts, (cx, cy), r, support)
    print(f"\n  wrote {os.path.join(outdir, 'welzl.svg')}")


def _svg(path, pts, centre, r, support, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Smallest enclosing circle (Welzl)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'blue = points, green = the minimum circle, red = the 2-3 support points that pin it down</text>',
    ]
    oy = 30

    def fx(x):
        return x

    def fy(y):
        return y + oy

    # the circle
    parts.append(f'<circle cx="{fx(centre[0]):.1f}" cy="{fy(centre[1]):.1f}" r="{r:.1f}" '
                 f'fill="#06d6a0" fill-opacity="0.08" stroke="#06d6a0" stroke-width="2"/>')
    parts.append(f'<circle cx="{fx(centre[0]):.1f}" cy="{fy(centre[1]):.1f}" r="3" fill="#06d6a0"/>')

    support_set = set(support)
    for p in pts:
        if p in support_set:
            continue
        parts.append(f'<circle cx="{fx(p[0]):.1f}" cy="{fy(p[1]):.1f}" r="3" fill="#4dabf7"/>')
    for p in support:
        parts.append(f'<circle cx="{fx(p[0]):.1f}" cy="{fy(p[1]):.1f}" r="5" fill="#ff6b6b"/>')

    parts.append(f'<text x="20" y="{height-16}" fill="#06d6a0" font-size="13">'
                 f'radius {r:.1f}, {len(support)} boundary points</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

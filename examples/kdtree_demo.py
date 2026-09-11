"""Demo: k-d tree -- fast nearest-neighbour search in space.

Builds a k-d tree over a 2D point cloud, checks nearest / k-nearest / radius queries against a
brute-force scan, and draws the cloud with a query point, its nearest neighbour, its k nearest,
and a radius query circle.

    python examples/kdtree_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kdtree import (KDTree, brute_nearest, brute_k_nearest,  # noqa: E402
                    brute_within_radius)


def _lcg_points(seed, n, hi=100.0):
    state = seed
    pts = []
    for _ in range(n):
        c = []
        for _ in range(2):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            c.append((state >> 8) / (1 << 24) * hi)
        pts.append((c[0], c[1]))
    return pts


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    pts = _lcg_points(1, 400)
    tree = KDTree(pts)
    print("k-d tree: nearest-neighbour search by recursive median splitting\n")
    print(f"  {len(pts)} points in 2D, tree height {tree.height()} (~log2 n = {math.log2(len(pts)):.1f})\n")

    q = (50.0, 50.0)
    near = tree.nearest(q)
    print(f"  nearest to {q}: {near[0]:.1f},{near[1]:.1f}  "
          f"(brute force agrees: {near == brute_nearest(pts, q)})")
    knn = tree.k_nearest(q, 5)
    print(f"  5 nearest agree with brute force: {knn == brute_k_nearest(pts, q, 5)}")
    r = 15.0
    within = tree.within_radius(q, r)
    print(f"  within radius {r:.0f}: {len(within)} points  "
          f"(brute force agrees: {sorted(within) == sorted(brute_within_radius(pts, q, r))})")

    # correctness sweep + a rough pruning estimate
    ok = all(tree.nearest(qq) == brute_nearest(pts, qq) for qq in _lcg_points(99, 500))
    print(f"\n  nearest matches brute force on 500 random queries: {ok}")
    print("  Each query descends to a leaf, then unwinds -- crossing the splitting plane only")
    print("  when the far side could hold something closer, so most branches are pruned.")
    print("  It powers k-NN classification, particle neighbour lists, and map/geographic search.")

    _svg(os.path.join(outdir, "kdtree.svg"), pts, q, near, knn, within, r)
    print(f"\n  wrote {os.path.join(outdir, 'kdtree.svg')}")


def _svg(path, pts, q, near, knn, within, radius, w=760, h=420):
    pad, span = 40, 100.0
    size = h - 90

    def SX(x):
        return pad + x / span * (size)

    def SY(y):
        return 70 + (span - y) / span * size  # flip y so it reads bottom-up

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'k-d tree: nearest, k-nearest, and radius queries</text>',
        f'<text x="20" y="46" fill="#8b949e" font-size="12">'
        f'query (yellow), nearest (green), 5-nearest (purple ring), radius query (orange circle)</text>',
        f'<rect x="{pad}" y="70" width="{size}" height="{size}" fill="none" '
        f'stroke="#21262d" stroke-width="1"/>',
    ]

    knn_set = set(knn)
    within_set = set(within)
    for p in pts:
        x, y = SX(p[0]), SY(p[1])
        if p in knn_set:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="none" '
                         f'stroke="#8338ec" stroke-width="2"/>')
        col = "#4dabf7" if p not in within_set else "#ff922b"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{col}"/>')

    # radius circle
    parts.append(f'<circle cx="{SX(q[0]):.1f}" cy="{SY(q[1]):.1f}" r="{radius / span * size:.1f}" '
                 f'fill="none" stroke="#ff922b" stroke-width="1.3" stroke-dasharray="4 3"/>')
    # nearest link
    parts.append(f'<line x1="{SX(q[0]):.1f}" y1="{SY(q[1]):.1f}" x2="{SX(near[0]):.1f}" '
                 f'y2="{SY(near[1]):.1f}" stroke="#06d6a0" stroke-width="1.8"/>')
    parts.append(f'<circle cx="{SX(near[0]):.1f}" cy="{SY(near[1]):.1f}" r="4" fill="#06d6a0"/>')
    # query point
    parts.append(f'<circle cx="{SX(q[0]):.1f}" cy="{SY(q[1]):.1f}" r="5" fill="#ffd43b"/>')
    parts.append(f'<text x="{SX(q[0]) + 8:.1f}" y="{SY(q[1]) - 8:.1f}" fill="#ffd43b" '
                 f'font-size="10">query</text>')

    # legend
    lx, ly = pad + size + 25, 90
    for col, lab in (("#ffd43b", "query point"), ("#06d6a0", "nearest neighbour"),
                     ("#8338ec", "5 nearest (ring)"), ("#ff922b", "within radius 15"),
                     ("#4dabf7", "other points")):
        parts.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="{col}"/>'
                     f'<text x="{lx + 12}" y="{ly + 4}" fill="#e6edf3" font-size="10">{lab}</text>')
        ly += 22

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: Delaunay triangulation and its dual Voronoi diagram.

Triangulates a random point set with Bowyer-Watson, extracts the dual Voronoi edges, and draws both
overlaid -- the Delaunay mesh in one colour and the Voronoi cell boundaries in another.

    python examples/delaunay_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from delaunay import (triangulate, voronoi_edges, circumcenter, delaunay_neighbors,
                      dedup_index_map)  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Delaunay triangulation and the dual Voronoi diagram\n")

    state = 424242

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    pts = [(40 + rng() * 320, 40 + rng() * 260) for _ in range(24)]
    uniq = dedup_index_map(pts)
    tris = triangulate(pts)
    vedges = voronoi_edges(pts, tris)
    adj = delaunay_neighbors(pts, tris)

    n = len(uniq)
    # verify the empty-circumcircle property live
    def empty_ok():
        for t in tris:
            c = circumcenter(uniq[t[0]], uniq[t[1]], uniq[t[2]])
            r = math.dist(c, uniq[t[0]])
            for k, p in enumerate(uniq):
                if k in t:
                    continue
                if math.dist(c, p) < r - 1e-9:
                    return False
        return True

    print(f"  {n} sites")
    print(f"  Delaunay triangles: {len(tris)}  (Euler: 2n-2-h)")
    print(f"  Voronoi edges (finite): {len(vedges)}")
    print(f"  empty-circumcircle property holds: {empty_ok()}")
    avg_deg = sum(len(v) for v in adj.values()) / n
    print(f"  average Delaunay degree: {avg_deg:.2f}")

    # nearest-neighbour cross-check: nearest site is always a Delaunay neighbour
    all_nn = True
    for i in range(n):
        best = min((j for j in range(n) if j != i), key=lambda j: math.dist(uniq[i], uniq[j]))
        if best not in adj[i]:
            all_nn = False
    print(f"  every site's nearest neighbour is a Delaunay edge: {all_nn}")

    print("\n  The Voronoi diagram partitions the plane into 'nearest-site' cells; its straight-line")
    print("  dual is the Delaunay triangulation, whose triangles have empty circumcircles and")
    print("  maximize the minimum angle. Voronoi vertices are the triangle circumcentres.")

    _svg(os.path.join(outdir, "delaunay.svg"), uniq, tris, vedges)
    print(f"\n  wrote {os.path.join(outdir, 'delaunay.svg')}")


def _svg(path, pts, tris, vedges, width=760, height=440):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Delaunay triangulation (blue) and its dual Voronoi diagram (orange)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'sites in green; Voronoi edges connect the circumcentres of adjacent Delaunay triangles</text>',
    ]

    def fx(x):
        return 20 + x

    def fy(y):
        return 70 + y

    # Voronoi edges (draw first, underneath)
    for (x0, y0), (x1, y1) in vedges:
        # clip wild circumcentres to the viewport-ish range so the SVG stays readable
        if all(-200 < v < 900 for v in (x0, y0, x1, y1)):
            parts.append(f'<line x1="{fx(x0):.1f}" y1="{fy(y0):.1f}" x2="{fx(x1):.1f}" '
                         f'y2="{fy(y1):.1f}" stroke="#ff922b" stroke-width="1.4"/>')

    # Delaunay edges
    seen = set()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            pa, pb = pts[a], pts[b]
            parts.append(f'<line x1="{fx(pa[0]):.1f}" y1="{fy(pa[1]):.1f}" x2="{fx(pb[0]):.1f}" '
                         f'y2="{fy(pb[1]):.1f}" stroke="#4dabf7" stroke-width="1.3"/>')

    # sites
    for x, y in pts:
        parts.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3.5" fill="#06d6a0"/>')

    parts.append(f'<text x="20" y="{height-16}" fill="#8b949e" font-size="12">'
                 f'{len(pts)} sites, {len(tris)} triangles -- every triangle circumcircle is empty</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

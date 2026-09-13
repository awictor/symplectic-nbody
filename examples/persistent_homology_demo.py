"""Demo: the H_0 persistence barcode of a point cloud with three clusters.

Samples a point cloud of three well-separated clusters, computes its zeroth persistent homology
barcode, and shows how the long bars reveal the cluster count while short bars are within-cluster
noise -- and that the finite bar deaths are exactly the minimum-spanning-tree edge weights. Draws the
point cloud and the barcode.

    python examples/persistent_homology_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from persistent_homology import (  # noqa: E402
    h0_barcode, finite_deaths, total_persistence, betti0_curve, mst_edge_weights,
)


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Persistent homology H_0: the barcode of a point cloud\n")

    rng = _lcg(2024)
    pts = []
    centers = [(0, 0), (60, 10), (30, 55)]
    for cx, cy in centers:
        for _ in range(8):
            pts.append((cx + (rng() - 0.5) * 12, cy + (rng() - 0.5) * 12))

    bars, merges = h0_barcode(pts)
    deaths = finite_deaths(bars)
    mst = mst_edge_weights(pts)

    print(f"  {len(pts)} points in {len(centers)} clusters")
    print(f"  barcode: {len(bars)} bars ({len(bars)-1} finite + 1 infinite)\n")

    deaths_sorted = sorted(deaths, reverse=True)
    print(f"  longest finite bars (deaths):")
    for d in deaths_sorted[:5]:
        kind = "inter-cluster" if d > 25 else "within-cluster"
        print(f"    death at eps = {d:6.2f}   ({kind})")

    n_long = sum(1 for d in deaths if d > 25)
    print(f"\n  {n_long} long bars + 1 infinite = {n_long + 1} robust components -> "
          f"{len(centers)} clusters recovered")
    print(f"  finite deaths == MST edge weights: "
          f"{all(abs(sorted(deaths)[i] - mst[i]) < 1e-9 for i in range(len(mst)))}")
    print(f"  total persistence (= MST weight): {total_persistence(bars):.2f}")

    _svg(os.path.join(outdir, "persistent_homology.svg"), pts, bars, deaths)
    print(f"\n  wrote {os.path.join(outdir, 'persistent_homology.svg')}")


def _svg(path, pts, bars, deaths, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="24" fill="#e6edf3" font-size="14">'
        f'Point cloud (left) and its H_0 persistence barcode (right); long bars = clusters</text>',
    ]

    # --- left: point cloud ---
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    lox, loy, lw, lh = 40, 50, 300, height - 100

    def sx(x):
        return lox + lw * (x - minx) / (maxx - minx + 1e-9)

    def sy(y):
        return loy + lh * (1 - (y - miny) / (maxy - miny + 1e-9))

    parts.append(f'<rect x="{lox}" y="{loy}" width="{lw}" height="{lh}" fill="none" stroke="#30363d"/>')
    for x, y in pts:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="#4dabf7"/>')

    # --- right: barcode ---
    box, boy, bw, bh = 400, 50, width - 440, height - 100
    finite = sorted((d for d in deaths), reverse=False)
    dmax = max(deaths) * 1.25 if deaths else 1
    n = len(bars)
    row_h = bh / n
    # sort bars: finite ascending by death, infinite last
    order = sorted(bars, key=lambda b: (b[1] if b[1] != math.inf else 1e18))

    def bx(eps):
        return box + bw * min(eps, dmax) / dmax

    parts.append(f'<rect x="{box}" y="{boy}" width="{bw}" height="{bh}" fill="none" stroke="#30363d"/>')
    for k, (birth, death) in enumerate(order):
        y = boy + k * row_h + row_h / 2
        if death == math.inf:
            x2 = box + bw
            col = "#06d6a0"
        else:
            x2 = bx(death)
            col = "#ffd43b" if death > 25 else "#8b949e"
        parts.append(f'<line x1="{bx(birth):.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
                     f'stroke="{col}" stroke-width="{max(row_h-2,1.5):.1f}"/>')
    parts.append(f'<text x="{box}" y="{boy+bh+16:.0f}" fill="#8b949e" font-size="9">eps = 0</text>')
    parts.append(f'<text x="{box+bw:.0f}" y="{boy+bh+16:.0f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">eps grows -&gt;</text>')
    parts.append(f'<text x="{box+bw-4:.0f}" y="{boy+12}" fill="#06d6a0" font-size="9" '
                 f'text-anchor="end">green = infinite (whole cloud)</text>')
    parts.append(f'<text x="{box+bw-4:.0f}" y="{boy+26}" fill="#ffd43b" font-size="9" '
                 f'text-anchor="end">yellow = inter-cluster</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

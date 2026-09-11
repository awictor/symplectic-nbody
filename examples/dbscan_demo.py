"""Demo: DBSCAN density clustering on shapes k-means cannot handle.

Clusters two interlocking half-moons plus a scatter of outliers, recovering the two crescents as
single clusters and flagging the outliers as noise -- something centroid methods split straight
through. Shows the core/border/noise breakdown and the k-distance graph whose elbow suggests eps.

    python examples/dbscan_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dbscan import dbscan, classify_points, n_clusters, k_distances, NOISE  # noqa: E402


def _data(seed=42):
    state = seed

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    X = []
    for _ in range(110):
        t = rng() * math.pi
        X.append([math.cos(t) + (rng() - 0.5) * 0.16,
                  math.sin(t) + (rng() - 0.5) * 0.16])
        X.append([1 - math.cos(t) + (rng() - 0.5) * 0.16,
                  -math.sin(t) + 0.4 + (rng() - 0.5) * 0.16])
    # sprinkle uniform outliers across the bounding area
    n_outliers = 12
    for _ in range(n_outliers):
        X.append([-1.3 + rng() * 3.6, -1.3 + rng() * 2.6])
    return X, n_outliers


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    X, n_out = _data()
    eps, min_pts = 0.22, 5
    labels = dbscan(X, eps, min_pts)
    kinds = classify_points(X, eps, min_pts)

    print("DBSCAN: density-based clustering of arbitrary shapes with noise\n")
    print(f"  {len(X)} points ({len(X)-n_out} on two interlocking moons + {n_out} outliers), "
          f"eps = {eps}, min_pts = {min_pts}\n")
    print(f"  clusters discovered (no k given): {n_clusters(labels)}")
    print(f"  points flagged as noise:          {labels.count(NOISE)}\n")

    print("  Point classification:")
    for kind in ("core", "border", "noise"):
        print(f"    {kind:>6}: {kinds.count(kind)}")

    sizes = {}
    for c in labels:
        if c != NOISE:
            sizes[c] = sizes.get(c, 0) + 1
    print("\n  Cluster sizes:")
    for c in sorted(sizes):
        print(f"    cluster {c}: {sizes[c]} points")

    print("\n  The k-distance graph (sorted distance to the 4th neighbour) -- its elbow marks eps:")
    kd = k_distances(X, min_pts - 1)
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9, 0.97):
        i = int(frac * (len(kd) - 1))
        bar = "#" * int(round(kd[i] / kd[-1] * 34))
        print(f"    {int(frac*100):>3}th pct: {kd[i]:.3f} {bar}")

    print("\n  Because it grows clusters by local density instead of distance to a centroid, DBSCAN")
    print("  recovers the crescents as single clusters (k-means would cut each in half), needs no k,")
    print("  and calls sparse points noise instead of forcing them into a cluster.")

    _svg(os.path.join(outdir, "dbscan.svg"), X, labels, kd)
    print(f"\n  wrote {os.path.join(outdir, 'dbscan.svg')}")


def _svg(path, X, labels, kd, width=760, height=430):
    palette = ["#4dabf7", "#ff6b6b", "#ffd43b", "#06d6a0", "#b197fc", "#ff922b"]
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    xa, xb = min(xs) - 0.2, max(xs) + 0.2
    ya, yb = min(ys) - 0.2, max(ys) + 0.2
    lx0, lx1 = 45, width // 2 - 10
    y0, y1 = height - 45, 65

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(v):
        return y0 - (v - ya) / (yb - ya) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'DBSCAN: two moons recovered by density; outliers flagged as noise</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: points coloured by cluster (grey x = noise); right: k-distance elbow for '
        f'choosing eps</text>',
    ]

    for i, (px, pv) in enumerate(X):
        c = labels[i]
        if c == NOISE:
            x, y = LX(px), LY(pv)
            parts.append(f'<line x1="{x-3:.1f}" y1="{y-3:.1f}" x2="{x+3:.1f}" y2="{y+3:.1f}" '
                         f'stroke="#8b949e" stroke-width="1.3"/>')
            parts.append(f'<line x1="{x-3:.1f}" y1="{y+3:.1f}" x2="{x+3:.1f}" y2="{y-3:.1f}" '
                         f'stroke="#8b949e" stroke-width="1.3"/>')
        else:
            parts.append(f'<circle cx="{LX(px):.1f}" cy="{LY(pv):.1f}" r="3.2" '
                         f'fill="{palette[c % len(palette)]}" opacity="0.9"/>')
    parts.append(f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.1"/>')

    # right: k-distance graph
    rx0, rx1 = width // 2 + 45, width - 30
    kmax = kd[-1]

    def RX(i):
        return rx0 + i / (len(kd) - 1) * (rx1 - rx0)

    def RY(v):
        return y0 - v / kmax * (y0 - y1)

    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.1"/>')
    pts = " ".join(f"{RX(i):.1f},{RY(kd[i]):.1f}" for i in range(len(kd)))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.2"/>')
    # mark the eps used
    parts.append(f'<line x1="{rx0}" y1="{RY(0.22):.1f}" x2="{rx1}" y2="{RY(0.22):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1.2" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{rx1-2:.1f}" y="{RY(0.22)-4:.1f}" fill="#ffd43b" font-size="9" '
                 f'text-anchor="end">eps = 0.22</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">sorted 4th-neighbour distance (elbow = eps)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

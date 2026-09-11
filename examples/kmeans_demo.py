"""Demo: k-means clustering -- finding groups in unlabelled data.

Clusters a set of blobs, showing the recovered groups and centroids, the inertia falling each
Lloyd iteration, and the "elbow" in inertia-vs-k that reveals the natural number of clusters.
Draws the coloured clusters with their centroids and the elbow curve.

    python examples/kmeans_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kmeans import kmeans, kmeans_best, inertia, cluster_sizes, silhouette, _Rng  # noqa: E402


def _blob(cx, cy, n, spread, seed):
    r = _Rng(seed)
    return [[cx + (r.random() - 0.5) * spread, cy + (r.random() - 0.5) * spread] for _ in range(n)]


DATA = (_blob(2, 2, 40, 2.0, 1) + _blob(9, 3, 40, 2.0, 2)
        + _blob(5, 9, 40, 2.0, 3) + _blob(12, 10, 40, 2.0, 4))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("k-means: partition points so each is nearest its cluster's centroid\n")
    c, labels, inr, it = kmeans(DATA, 4, seed=1)
    print(f"  {len(DATA)} points, k=4: converged in {it} iterations")
    print(f"  cluster sizes {cluster_sizes(labels, 4)}, inertia {inr:.1f}, "
          f"silhouette {silhouette(DATA, labels, 4):.3f}")
    print(f"  centroids: {[[round(v, 1) for v in ct] for ct in c]}\n")

    print("  Inertia falls every Lloyd iteration (assign, then re-centre):")
    prev = None
    for mi in range(1, 7):
        _, _, i_mi, _ = kmeans(DATA, 4, seed=1, max_iter=mi)
        drop = "" if prev is None else f"  (-{prev - i_mi:.1f})"
        print(f"    after {mi} iter(s): inertia {i_mi:.1f}{drop}")
        prev = i_mi

    print("\n  The 'elbow': inertia vs k drops sharply until the true cluster count, then flattens.")
    print(f"  {'k':>4}{'inertia':>12}{'silhouette':>12}")
    for k in range(1, 8):
        _, lab, ik = kmeans_best(DATA, k, restarts=5)
        sil = silhouette(DATA, lab, k) if k >= 2 else 0.0
        print(f"  {k:>4}{ik:>12.1f}{sil:>12.3f}")
    print("\n  The elbow at k=4 and the peak silhouette both flag the true four groups. k-means")
    print("  powers customer segmentation, color quantization, and vector quantization -- and")
    print("  k-means++ seeding spreads the initial centres to avoid Lloyd's bad local minima.")

    _svg(os.path.join(outdir, "kmeans.svg"), c, labels)
    print(f"\n  wrote {os.path.join(outdir, 'kmeans.svg')}")


def _svg(path, centroids, labels, w=760, h=400):
    xs = [p[0] for p in DATA]
    ys = [p[1] for p in DATA]
    lo = min(min(xs), min(ys)) - 1
    hi = max(max(xs), max(ys)) + 1

    # left: clustered scatter
    lx0, lx1 = 45, w // 2 - 10
    y0, y1 = h - 45, 65

    def LX(x):
        return lx0 + (x - lo) / (hi - lo) * (lx1 - lx0)

    def LY(y):
        return y0 - (y - lo) / (hi - lo) * (y0 - y1)

    colors = ["#4dabf7", "#06d6a0", "#ff922b", "#ff6b6b", "#8338ec", "#ffd43b", "#b197fc"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'k-means: points coloured by cluster, X marks the centroids</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'four recovered clusters (left); the inertia elbow at the true k (right)</text>',
    ]
    for i, p in enumerate(DATA):
        col = colors[labels[i] % len(colors)]
        parts.append(f'<circle cx="{LX(p[0]):.1f}" cy="{LY(p[1]):.1f}" r="2.5" fill="{col}" opacity="0.7"/>')
    for c, ct in enumerate(centroids):
        x, y = LX(ct[0]), LY(ct[1])
        col = colors[c % len(colors)]
        parts.append(f'<text x="{x:.1f}" y="{y+5:.1f}" fill="{col}" font-size="18" '
                     f'font-weight="bold" text-anchor="middle" stroke="#0d1117" stroke-width="0.6">X</text>')

    # right: inertia elbow (inertia vs k)
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 55, 70
    ks = list(range(1, 8))
    inertias = [kmeans_best(DATA, k, restarts=5)[2] for k in ks]
    imax = max(inertias)

    def RX(k):
        return rx0 + (k - ks[0]) / (ks[-1] - ks[0]) * (rx1 - rx0)

    def RY(v):
        return ry0 - v / imax * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = " ".join(f"{RX(k):.1f},{RY(v):.1f}" for k, v in zip(ks, inertias))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    for k, v in zip(ks, inertias):
        r = 4 if k == 4 else 2.5
        col = "#ffd43b" if k == 4 else "#06d6a0"
        parts.append(f'<circle cx="{RX(k):.1f}" cy="{RY(v):.1f}" r="{r}" fill="{col}"/>')
        parts.append(f'<text x="{RX(k):.1f}" y="{ry0+15:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{RX(4):.1f}" y="{RY(inertias[3])-8:.1f}" fill="#ffd43b" font-size="9" '
                 f'text-anchor="middle">elbow at k=4</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">number of clusters k -> inertia</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

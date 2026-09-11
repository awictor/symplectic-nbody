"""Demo: t-SNE embedding high-dimensional clusters into a 2-D map.

Generates well-separated clusters in high dimensions, embeds them with t-SNE, reports the KL
divergence falling and the neighbourhood-preservation (trustworthiness) score, and draws the 2-D map
coloured by cluster.

    python examples/tsne_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tsne import tsne, trustworthiness  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("t-SNE: nonlinear dimensionality reduction that preserves neighborhoods\n")

    state = 7

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # 4 clusters in 8 dimensions
    dim = 8
    centers = []
    for c in range(4):
        ctr = [0.0] * dim
        # each cluster sits far along two random axes
        a, b = c * 2 % dim, (c * 2 + 1) % dim
        ctr[a] = 12.0
        ctr[b] = 12.0
        centers.append(ctr)

    X, labels = [], []
    per = 20
    for c, ctr in enumerate(centers):
        for _ in range(per):
            X.append([ctr[k] + (rng() * 2 - 1) * 1.5 for k in range(dim)])
            labels.append(c)

    print(f"  {len(X)} points in {dim} dimensions, {len(centers)} true clusters")

    Y, hist = tsne(X, perplexity=15, n_iter=500, return_history=True, seed=1)

    print(f"  KL divergence: {hist[0]:.3f} (start) -> {hist[-1]:.3f} (end)")
    print(f"  trustworthiness (k=8): {trustworthiness(X, Y, k=8):.3f}  (1.0 = perfect neighbor preservation)")

    # cluster separation ratio
    intra, inter = [], []
    for i in range(len(Y)):
        for j in range(i + 1, len(Y)):
            d = math.hypot(Y[i][0] - Y[j][0], Y[i][1] - Y[j][1])
            (intra if labels[i] == labels[j] else inter).append(d)
    ratio = (sum(inter) / len(inter)) / (sum(intra) / len(intra))
    print(f"  2-D separation: inter-cluster distance is {ratio:.1f}x the intra-cluster distance")

    print("\n  t-SNE matches per-point Gaussian neighbor probabilities in high-D to a heavy-tailed")
    print("  Student-t kernel in 2-D, minimizing KL(P||Q) by gradient descent. The heavy tail cures")
    print("  the crowding problem, letting clusters breathe apart -- which is why the map is so clear.")

    _svg(os.path.join(outdir, "tsne.svg"), Y, labels, hist)
    print(f"\n  wrote {os.path.join(outdir, 'tsne.svg')}")


def _svg(path, Y, labels, hist, width=760, height=440):
    colors = ["#4dabf7", "#ffd43b", "#ff6b6b", "#06d6a0", "#b197fc", "#ff922b"]

    # left panel: the 2-D embedding; right strip: KL curve
    map_w = 480
    xs = [p[0] for p in Y]
    ys = [p[1] for p in Y]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 40

    def mx(x):
        return pad + (x - xmin) / (xmax - xmin + 1e-12) * (map_w - 2 * pad)

    def my(y):
        return 70 + (height - 110) - (y - ymin) / (ymax - ymin + 1e-12) * (height - 180)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f't-SNE map of 8-D clusters (colour = true cluster)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'left: 2-D embedding; right: KL(P||Q) falling over training</text>',
    ]

    for (x, y), lab in zip(Y, labels):
        parts.append(f'<circle cx="{mx(x):.1f}" cy="{my(y):.1f}" r="4" '
                     f'fill="{colors[lab % len(colors)]}" fill-opacity="0.85"/>')

    # KL curve on the right
    cx0, cy0, cw, ch = map_w + 40, 90, width - map_w - 80, height - 160
    kmax = max(hist)
    kmin = min(hist)
    parts.append(f'<line x1="{cx0}" y1="{cy0+ch}" x2="{cx0+cw}" y2="{cy0+ch}" stroke="#484f58"/>')
    parts.append(f'<line x1="{cx0}" y1="{cy0}" x2="{cx0}" y2="{cy0+ch}" stroke="#484f58"/>')
    pts = " ".join(f"{cx0 + i/len(hist)*cw:.1f},{cy0 + ch - (h-kmin)/(kmax-kmin+1e-12)*ch:.1f}"
                   for i, h in enumerate(hist))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="1.8"/>')
    parts.append(f'<text x="{cx0}" y="{cy0-8}" fill="#8b949e" font-size="11">KL divergence</text>')
    parts.append(f'<text x="{cx0+cw:.0f}" y="{cy0+ch+18}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">iterations</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

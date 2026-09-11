"""Demo: spectral clustering of shapes k-means cannot separate.

Clusters two concentric rings by the eigenvectors of the affinity-graph Laplacian, showing that
k-means fails on the raw coordinates but succeeds trivially in the spectral embedding -- where the
tangled rings become two tight, separable blobs. Also shows the Laplacian's near-zero eigenvalues
counting the clusters.

    python examples/spectral_clustering_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectral_clustering import (spectral_clustering, affinity_matrix,  # noqa: E402
                                 laplacian, smallest_eigenvectors, count_components)
from kmeans import kmeans_best


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 42

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # two concentric rings -- the textbook case k-means gets wrong
    X, truth = [], []
    for _ in range(22):
        t = rng() * 2 * math.pi
        X.append([math.cos(t) + (rng() - 0.5) * 0.08, math.sin(t) + (rng() - 0.5) * 0.08])
        truth.append(0)
        X.append([3 * math.cos(t) + (rng() - 0.5) * 0.08, 3 * math.sin(t) + (rng() - 0.5) * 0.08])
        truth.append(1)

    labels, vals = spectral_clustering(X, 2, sigma=0.4, seed=1)
    _, km_labels, _ = kmeans_best(X, 2, seed=1)

    def purity(lab):
        for c in set(lab):
            ts = [truth[i] for i in range(len(lab)) if lab[i] == c]
            if max(ts.count(0), ts.count(1)) / len(ts) < 0.9:
                return False
        return True

    print("Spectral clustering: cutting a graph by its Laplacian eigenvectors\n")
    print(f"  {len(X)} points on two concentric rings (inner radius 1, outer 3)\n")
    print(f"  spectral clustering separates the rings: {purity(labels)}")
    print(f"  plain k-means separates the rings:       {purity(km_labels)}  (it can't -- "
          f"centroids can't wrap a ring)\n")

    print(f"  Laplacian's smallest eigenvalues: {[round(v, 4) for v in vals]}")
    print("  Two near-zero values => two clusters; the theorem: zero-eigenvalue multiplicity")
    print("  equals the number of connected components of the affinity graph.\n")

    # show the spectral embedding turns tangled rings into separable blobs
    W = affinity_matrix(X, sigma=0.4)
    L = laplacian(W, normalized=True)
    _, vecs = smallest_eigenvectors(L, 2)
    emb = [[vecs[0][i], vecs[1][i]] for i in range(len(X))]
    # separation in embedding vs raw coordinates: ratio of between- to within-cluster spread
    print("  In the 2-eigenvector embedding the rings collapse to two tight point clouds --")
    print("  k-means in that space is trivial. That is the whole trick: solve a hard geometric")
    print("  clustering by an easy one after a spectral change of coordinates.\n")

    # component-count demonstration on separated blobs
    comp = []
    for cx, cy in [(0.0, 0.0), (15.0, 0.0), (0.0, 15.0)]:
        for _ in range(6):
            comp.append([cx + (rng() - 0.5), cy + (rng() - 0.5)])
    Wc = affinity_matrix(comp, sigma=1.0)
    print(f"  Sanity check: 3 far-apart blobs -> {count_components(Wc, tol=1e-3)} graph components "
          f"(and 3 zero Laplacian eigenvalues).")

    _svg(os.path.join(outdir, "spectral_clustering.svg"), X, labels, km_labels, emb, vals)
    print(f"\n  wrote {os.path.join(outdir, 'spectral_clustering.svg')}")


def _svg(path, X, labels, km_labels, emb, vals, width=760, height=440):
    palette = ["#4dabf7", "#ff6b6b"]
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    xa, xb = min(xs) - 0.4, max(xs) + 0.4
    ya, yb = min(ys) - 0.4, max(ys) + 0.4

    def panel_scatter(x0, x1, lab, title, XX, ax, bx, ay, by):
        py0, py1 = height - 50, 78
        cells = []

        def PX(v):
            return x0 + (v - ax) / (bx - ax) * (x1 - x0)

        def PY(v):
            return py0 - (v - ay) / (by - ay) * (py0 - py1)

        for i, (px, pv) in enumerate(XX):
            cells.append(f'<circle cx="{PX(px):.1f}" cy="{PY(pv):.1f}" r="3.2" '
                         f'fill="{palette[lab[i] % 2]}" opacity="0.9"/>')
        cells.append(f'<text x="{(x0+x1)/2:.1f}" y="{py1-8:.1f}" fill="#e6edf3" font-size="12" '
                     f'text-anchor="middle">{title}</text>')
        return cells

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Spectral clustering: rings split correctly (left), the embedding that does it '
        f'(right)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: original points coloured by spectral cluster; right: the two Laplacian '
        f'eigenvectors -- tangled rings become separable blobs</text>',
    ]

    mid = width // 2
    parts += panel_scatter(45, mid - 15, labels, "rings, spectral labels", X, xa, xb, ya, yb)

    # right: embedding scatter, coloured by the same labels
    exs = [e[0] for e in emb]
    eys = [e[1] for e in emb]
    ea, eb = min(exs) - 0.02, max(exs) + 0.02
    fa, fb = min(eys) - 0.02, max(eys) + 0.02
    parts += panel_scatter(mid + 15, width - 20, labels, "eigenvector embedding", emb, ea, eb, fa, fb)

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

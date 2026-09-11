"""Demo: k-nearest-neighbours and the bias-variance effect of k.

Classifies a noisy two-class problem and shows how the decision boundary goes from jagged (k=1,
overfitting) to smooth (large k, high bias) as k grows, with leave-one-out cross-validation picking
the sweet spot. Also fits a k-NN regression to a noisy sine.

    python examples/knn_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knn import KNN, loo_cross_val  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 99

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # two classes split by a diagonal, with label noise near the boundary
    X, y = [], []
    for _ in range(120):
        a, b = rng() * 6, rng() * 6
        label = 1 if a + b > 6 else 0
        if rng() < 0.15:
            label = 1 - label
        X.append([a, b])
        y.append(label)

    print("k-nearest-neighbours: lazy, non-parametric classification\n")
    print(f"  {len(X)} points, 2 classes split by a diagonal with 15% label noise\n")

    print("  Training accuracy vs k (k=1 memorizes, larger k smooths):")
    for k in (1, 3, 7, 15, 31):
        acc = KNN(k=k).fit(X, y).score(X, y)
        print(f"    k = {k:>2}: train accuracy {acc:.1%}")

    best_k, scores = loo_cross_val(X, y, [1, 3, 5, 7, 9, 11, 15, 21, 31])
    print("\n  Leave-one-out cross-validation (honest accuracy) picks k:")
    for k in sorted(scores):
        mark = "  <- best" if k == best_k else ""
        bar = "#" * int(round(scores[k] * 40))
        print(f"    k = {k:>2}: LOO {scores[k]:.3f} {bar}{mark}")
    print(f"\n  Best k by LOO = {best_k}: k=1 overfits the noise, very large k oversmooths.\n")

    # regression on a noisy sine
    Xr = [[i * 0.25] for i in range(40)]
    yr = [math.sin(i * 0.25) + (rng() - 0.5) * 0.3 for i in range(40)]
    kr_u = KNN(k=5, task="regress", weights="uniform").fit(Xr, yr)
    kr_w = KNN(k=5, task="regress", weights="distance").fit(Xr, yr)
    print(f"  k-NN regression on a noisy sine (k=5): R^2 uniform {kr_u.score(Xr, yr):.3f}, "
          f"distance-weighted {kr_w.score(Xr, yr):.3f}")

    print("\n  k-NN builds no model at all -- it just stores the data and votes among the nearest")
    print("  neighbours at query time. Small k means low bias but high variance (jagged, noise-")
    print("  fitting); large k means the opposite. Cross-validation finds the balance.")

    _svg(os.path.join(outdir, "knn.svg"), X, y, best_k, scores)
    print(f"\n  wrote {os.path.join(outdir, 'knn.svg')}")


def _svg(path, X, y, best_k, scores, width=760, height=440):
    palette = ["#4dabf7", "#ff6b6b"]
    fills = ["#16324f", "#3d1d1d"]
    xa, xb, ya, yb = 0.0, 6.0, 0.0, 6.0

    def panel(x0, x1, k, title):
        model = KNN(k=k).fit(X, y)
        gx, gy = 54, 54
        cw = (x1 - x0) / gx
        py0, py1 = height - 55, 78
        ch = (py0 - py1) / gy
        cells = []
        # decision regions
        for i in range(gx):
            for j in range(gy):
                xv = xa + (i + 0.5) / gx * (xb - xa)
                yv = ya + (j + 0.5) / gy * (yb - ya)
                cls = model.predict([[xv, yv]])[0]
                cells.append(f'<rect x="{x0 + i*cw:.1f}" y="{py0 - (j+1)*ch:.1f}" '
                             f'width="{cw+0.6:.1f}" height="{ch+0.6:.1f}" fill="{fills[cls]}"/>')

        def PX(v):
            return x0 + (v - xa) / (xb - xa) * (x1 - x0)

        def PY(v):
            return py0 - (v - ya) / (yb - ya) * (py0 - py1)

        for idx, (px, pv) in enumerate(X):
            cells.append(f'<circle cx="{PX(px):.1f}" cy="{PY(pv):.1f}" r="2.4" '
                         f'fill="{palette[y[idx]]}" stroke="#0d1117" stroke-width="0.4"/>')
        cells.append(f'<text x="{(x0+x1)/2:.1f}" y="{py1-8:.1f}" fill="#e6edf3" font-size="12" '
                     f'text-anchor="middle">{title}</text>')
        return cells

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'k-NN: the decision boundary smooths as k grows</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'k=1 traces every noisy point (jagged); larger k averages over neighbours (smooth)</text>',
    ]
    mid = width // 2
    parts += panel(45, mid - 15, 1, "k = 1 (overfits)")
    parts += panel(mid + 15, width - 20, best_k, f"k = {best_k} (LOO-selected)")

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

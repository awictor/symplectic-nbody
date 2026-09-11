"""Demo: a random-forest classifier vs a single decision tree.

Trains on a noisy circular decision region with irrelevant features, and shows the forest
generalizing better than a single overfit tree, its out-of-bag score tracking true test error, and
feature importances flagging the two real features. Draws the smoother forest decision boundary
beside the jagged single-tree one.

    python examples/random_forest_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from random_forest import RandomForest  # noqa: E402
from decision_tree import DecisionTree  # noqa: E402


def _data(n, seed=999, noise_features=3, flip=0.10):
    state = seed

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    X, y = [], []
    for _ in range(n):
        a, b = rng() * 10, rng() * 10
        extra = [rng() * 10 for _ in range(noise_features)]
        label = 1 if (a - 5) ** 2 + (b - 5) ** 2 < 9 else 0
        if rng() < flip:
            label = 1 - label
        X.append([a, b] + extra)
        y.append(label)
    return X, y


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    X, y = _data(500)
    Xtr, ytr, Xte, yte = X[:350], y[:350], X[350:], y[350:]

    tree = DecisionTree().fit(Xtr, ytr)                       # unlimited -> overfits
    forest = RandomForest(n_trees=41, max_depth=8, seed=1).fit(Xtr, ytr)

    print("Random forest: a committee of decorrelated decision trees\n")
    print(f"  {len(Xtr)} train / {len(Xte)} test points, 2 real + 3 noise features, "
          f"10% label flips\n")
    print(f"  single tree (unlimited depth):  train {tree.accuracy(Xtr, ytr):.1%}, "
          f"test {tree.accuracy(Xte, yte):.1%},  {tree.n_leaves()} leaves (memorized noise)")
    print(f"  random forest (41 trees):       train {forest.accuracy(Xtr, ytr):.1%}, "
          f"test {forest.accuracy(Xte, yte):.1%},  out-of-bag {forest.oob_score_:.1%}\n")

    print("  Out-of-bag error is a free validation estimate -- each row is scored only by the")
    print("  ~37% of trees that never trained on it. Here OOB tracks true test error closely.\n")

    imp = forest.feature_importances()
    print("  Feature importances (real features 0,1 should dominate the 3 noise features):")
    names = ["x", "y", "noise1", "noise2", "noise3"]
    for i, v in enumerate(imp):
        bar = "#" * int(round(v * 40))
        print(f"    {names[i]:>7}: {v:5.2f} {bar}")

    print("\n  More trees only help -- test accuracy stabilizes as the ensemble grows:")
    for nt in (1, 3, 9, 21, 41):
        f = RandomForest(n_trees=nt, max_depth=8, seed=1).fit(Xtr, ytr)
        print(f"    {nt:>3} trees: test {f.accuracy(Xte, yte):.1%}, oob {f.oob_score_:.1%}")

    print("\n  Bagging (bootstrap rows) plus per-split feature subsampling decorrelate the trees")
    print("  so their errors cancel while signal adds. The forest boundary is smoother and more")
    print("  robust than any single tree -- the strong default for tabular data.")

    _svg(os.path.join(outdir, "random_forest.svg"), Xtr, ytr, tree, forest)
    print(f"\n  wrote {os.path.join(outdir, 'random_forest.svg')}")


def _svg(path, X, y, tree, forest, width=760, height=430):
    colors = ["#4dabf7", "#ff6b6b"]        # class 0, 1 points
    fills = ["#16324f", "#3d1d1d"]         # region tints
    pt = [[row[0], row[1]] for row in X]   # plot only the 2 real features
    xa, xb, ya, yb = 0.0, 10.0, 0.0, 10.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Random forest vs single tree: voting smooths the decision boundary</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'noisy circular rule; the single tree (left) memorizes noise, the forest (right) '
        f'generalizes</text>',
    ]

    def panel(x0, x1, model, title, pad_feats):
        gx, gy = 70, 60
        cw = (x1 - x0) / gx
        py0, py1 = height - 45, 65
        ch = (py0 - py1) / gy
        cells = []
        for i in range(gx):
            for j in range(gy):
                xv = xa + (i + 0.5) / gx * (xb - xa)
                yv = ya + (j + 0.5) / gy * (yb - ya)
                cls = model.predict([[xv, yv] + pad_feats])[0]
                cx = x0 + i * cw
                cy = py0 - (j + 1) * ch
                cells.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw+0.6:.1f}" '
                             f'height="{ch+0.6:.1f}" fill="{fills[cls % 2]}"/>')
        # points
        for (px, pv), label in zip(pt, y):
            cx = x0 + (px - xa) / (xb - xa) * (x1 - x0)
            cy = py0 - (pv - ya) / (yb - ya) * (py0 - py1)
            cells.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.6" '
                         f'fill="{colors[label % 2]}" stroke="#0d1117" stroke-width="0.5"/>')
        cells.append(f'<text x="{(x0+x1)/2:.1f}" y="{py0+18:.1f}" fill="#8b949e" '
                     f'font-size="11" text-anchor="middle">{title}</text>')
        return cells

    mid = width // 2
    # noise features held at their mid value (5.0) for the 2-D slice
    parts += panel(45, mid - 15, tree, "single tree (overfit)", [5.0, 5.0, 5.0])
    parts += panel(mid + 15, width - 20, forest, "random forest (41 trees)", [5.0, 5.0, 5.0])

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

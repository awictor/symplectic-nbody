"""Demo: a CART decision-tree classifier.

Grows a tree on three well-separated 2-D blobs, prints the learned if/else rules and feature
importances, and shows how a depth cap trades fit for simplicity. Draws the axis-aligned decision
regions the tree carves out of the plane, with the training points on top.

    python examples/decision_tree_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision_tree import DecisionTree, gini, entropy  # noqa: E402


def _blobs(seed=12345):
    """Three 2-D Gaussian-ish blobs (LCG, high bits), labels 0/1/2."""
    state = seed

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    X, y = [], []
    for cls, (cx, cy) in enumerate([(2.0, 2.0), (8.0, 8.0), (2.0, 8.0)]):
        for _ in range(30):
            X.append([cx + (rng() - 0.5) * 2.5, cy + (rng() - 0.5) * 2.5])
            y.append(cls)
    return X, y


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    X, y = _blobs()
    tree = DecisionTree(criterion="gini").fit(X, y)

    print("Decision tree (CART): recursive best-split classification\n")
    print(f"  {len(X)} points, 3 classes, 2 features (x, y)")
    print(f"  training accuracy {tree.accuracy(X, y):.1%}, "
          f"depth {tree.depth()}, {tree.n_leaves()} leaves\n")

    print("  Learned rules (Gini impurity):")
    for line in tree.rules():
        print("    " + line)

    imp = tree.feature_importances()
    print(f"\n  Feature importances:  x = {imp[0]:.2f},  y = {imp[1]:.2f}")

    print("\n  Depth cap trades fit for simplicity:")
    for d in (1, 2, 3, None):
        t = DecisionTree(max_depth=d).fit(X, y)
        tag = "unlimited" if d is None else f"max_depth={d}"
        print(f"    {tag:>12}: {t.n_leaves():>2} leaves, accuracy {t.accuracy(X, y):.1%}")

    # entropy vs gini pick the same clean splits here
    te = DecisionTree(criterion="entropy").fit(X, y)
    print(f"\n  Entropy criterion: accuracy {te.accuracy(X, y):.1%}, {te.n_leaves()} leaves")
    print(f"  (a pure node has gini 0 and entropy 0; a 50/50 node has gini "
          f"{gini([0,1]):.2f}, entropy {entropy([0,1]):.2f})")

    print("\n  A tree asks the single threshold question that most purifies its samples,")
    print("  then recurses. The result is a set of axis-aligned rectangles tiling the plane,")
    print("  each labelled by majority vote -- readable rules, the base learner of forests.")

    _svg(os.path.join(outdir, "decision_tree.svg"), X, y, tree)
    print(f"\n  wrote {os.path.join(outdir, 'decision_tree.svg')}")


def _svg(path, X, y, tree, width=760, height=420):
    colors = ["#4dabf7", "#ffd43b", "#ff6b6b"]      # class 0, 1, 2 points
    fills = ["#16324f", "#3d3410", "#3d1d1d"]       # muted region tints
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    xa, xb = min(xs) - 0.5, max(xs) + 0.5
    ya, yb = min(ys) - 0.5, max(ys) + 0.5
    px0, px1 = 45, width - 240
    py0, py1 = height - 45, 65

    def PX(x):
        return px0 + (x - xa) / (xb - xa) * (px1 - px0)

    def PY(v):
        return py0 - (v - ya) / (yb - ya) * (py0 - py1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Decision tree: axis-aligned regions carved by best-split recursion</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'each rectangle is a leaf; colour = predicted class; dots = training points</text>',
    ]

    # paint the decision regions by sampling a grid and drawing cells
    gx, gy = 90, 60
    cw = (px1 - px0) / gx
    ch = (py0 - py1) / gy
    for i in range(gx):
        for j in range(gy):
            xv = xa + (i + 0.5) / gx * (xb - xa)
            yv = ya + (j + 0.5) / gy * (yb - ya)
            cls = tree.predict([[xv, yv]])[0]
            cx = px0 + i * cw
            cy = py0 - (j + 1) * ch
            parts.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw+0.6:.1f}" '
                         f'height="{ch+0.6:.1f}" fill="{fills[cls % 3]}"/>')

    # axes
    parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}" stroke="#8b949e" stroke-width="1.2"/>')

    # training points
    for (x, v), label in zip(X, y):
        parts.append(f'<circle cx="{PX(x):.1f}" cy="{PY(v):.1f}" r="3.2" '
                     f'fill="{colors[label % 3]}" stroke="#0d1117" stroke-width="0.6"/>')

    # rules panel on the right
    tx = px1 + 20
    parts.append(f'<text x="{tx}" y="72" fill="#e6edf3" font-size="13">learned rules</text>')
    ln = 92
    for line in tree.rules()[:14]:
        depth = (len(line) - len(line.lstrip())) // 2
        txt = line.strip()
        col = "#06d6a0" if txt.startswith("predict") else "#b197fc"
        parts.append(f'<text x="{tx + depth*10}" y="{ln}" fill="{col}" font-size="10">'
                     f'{_esc(txt[:30])}</text>')
        ln += 15

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()

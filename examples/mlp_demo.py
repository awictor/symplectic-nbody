"""Demo: a multi-layer perceptron learning XOR and a nonlinear boundary via backprop.

Trains an MLP on XOR (which no linear model can solve), fits a nonlinear regression, and learns a
circular decision boundary. Draws the XOR loss curve and the learned decision surface.

    python examples/mlp_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlp import MLP  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Multi-layer perceptron + backpropagation\n")

    # XOR
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]
    net = MLP([2, 4, 1], activation="tanh", output_activation="sigmoid", seed=3)
    hist = net.train(X, Y, epochs=3000, lr=0.5)
    print("  XOR (not linearly separable):")
    for x, y in zip(X, Y):
        print(f"    {x} -> {net.predict(x)[0]:.3f}  (target {y[0]})")
    print(f"    loss {hist[0]:.4f} -> {hist[-1]:.6f} over 3000 epochs")

    # a linear model for contrast
    lin = MLP([2, 1], activation="identity", output_activation="sigmoid", seed=1)
    lin.train(X, Y, epochs=2000, lr=0.5)
    lin_correct = sum((lin.predict(X[i])[0] > 0.5) == bool(Y[i][0]) for i in range(4))
    print(f"    a linear model gets only {lin_correct}/4 right (XOR needs a hidden layer)")

    # circular decision boundary: inside a circle vs outside
    state = 20260911

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    Xc, Yc = [], []
    for _ in range(400):
        x, y = rng() * 4 - 2, rng() * 4 - 2
        Xc.append([x, y])
        Yc.append([1 if x * x + y * y < 1.5 else 0])
    clf = MLP([2, 10, 6, 1], activation="tanh", output_activation="sigmoid", seed=5)
    clf.train(Xc, Yc, epochs=800, lr=0.1)
    acc = sum((clf.predict(Xc[i])[0] > 0.5) == bool(Yc[i][0]) for i in range(400)) / 400
    print(f"\n  Circular decision boundary (inside a disk): accuracy {acc:.3f}")
    print("    a linear boundary could never separate a disk from its surroundings")

    print("\n  Backpropagation is the chain rule run backward: a forward pass computes the loss, then")
    print("  the gradient flows back layer by layer, each reusing the gradient handed down from")
    print("  above, so the whole gradient costs one backward pass. Gradient descent does the rest.")

    _svg(os.path.join(outdir, "mlp.svg"), hist, clf)
    print(f"\n  wrote {os.path.join(outdir, 'mlp.svg')}")


def _svg(path, hist, clf, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'MLP: XOR loss curve (left) and a learned circular boundary (right)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'left: training loss falling; right: the network output over the plane (blue=inside, dark=outside)</text>',
    ]

    # left: loss curve (log scale)
    lx0, ly0, lw, lh = 60, 90, 300, 290
    parts.append(f'<line x1="{lx0}" y1="{ly0+lh}" x2="{lx0+lw}" y2="{ly0+lh}" stroke="#484f58"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly0+lh}" stroke="#484f58"/>')
    lmax = math.log10(max(hist))
    lmin = math.log10(max(min(hist), 1e-7))
    n = len(hist)
    pts = []
    for i, v in enumerate(hist):
        px = lx0 + i / n * lw
        py = ly0 + lh - (math.log10(max(v, 1e-7)) - lmin) / (lmax - lmin) * lh
        pts.append(f"{px:.1f},{py:.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="1.8"/>')
    parts.append(f'<text x="{lx0}" y="{ly0-6}" fill="#8b949e" font-size="11">XOR loss (log scale) vs epoch</text>')

    # right: decision surface as a colored grid
    gx0, gy0, gs = 420, 90, 290
    grid = 40
    cell = gs / grid
    for gi in range(grid):
        for gj in range(grid):
            x = -2 + 4 * gi / grid
            y = -2 + 4 * gj / grid
            p = clf.predict([x, y])[0]
            shade = int(30 + p * 150)
            r, g, b = int(20 + p * 60), int(40 + p * 130), int(60 + p * 190)
            px = gx0 + gi * cell
            py = gy0 + gs - (gj + 1) * cell
            parts.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cell+0.5:.1f}" '
                         f'height="{cell+0.5:.1f}" fill="rgb({r},{g},{b})"/>')
    # true circle boundary
    cxp = gx0 + gs / 2
    cyp = gy0 + gs / 2
    parts.append(f'<circle cx="{cxp:.1f}" cy="{cyp:.1f}" r="{math.sqrt(1.5)/4*gs:.1f}" '
                 f'fill="none" stroke="#ffd43b" stroke-width="1.5" stroke-dasharray="4,3"/>')
    parts.append(f'<text x="{gx0}" y="{gy0-6}" fill="#8b949e" font-size="11">'
                 f'learned output over the plane (yellow = true circle)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

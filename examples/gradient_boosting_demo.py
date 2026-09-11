"""Demo: gradient boosting for regression and classification.

Fits a noisy nonlinear curve by adding shallow trees one at a time -- showing the fit sharpen and
the training loss fall monotonically -- and boosts a classifier on a circular boundary. Contrasts
the ensemble with a single tree and shows the learning-rate / n-trees trade.

    python examples/gradient_boosting_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gradient_boosting import (GradientBoostingRegressor,  # noqa: E402
                               GradientBoostingClassifier, _RegTree)


def _target(x):
    return math.sin(x) + 0.3 * x


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    state = 17

    def rng():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 16) / 65536.0

    # noisy nonlinear regression target
    X = [[i * 0.12] for i in range(70)]
    y = [_target(i * 0.12) + (rng() - 0.5) * 0.4 for i in range(70)]

    gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.1, max_depth=3).fit(X, y)

    print("Gradient boosting: shallow trees added in sequence, each fixing the last's errors\n")
    print(f"  regression on {len(X)} noisy points of sin(x) + 0.3x\n")
    print(f"  final R^2 {gbr.r_squared(X, y):.4f}, MSE {gbr.mse(X, y):.4f}")

    staged = gbr.staged_predict(X)
    losses = [sum((y[i] - p[i]) ** 2 for i in range(len(y))) / len(y) for p in staged]
    print("\n  Training loss falls monotonically as trees are added:")
    for t in (0, 1, 4, 9, 29, 59, 119):
        if t < len(losses):
            print(f"    after {t+1:>3} trees: MSE {losses[t]:.4f}")

    single = _RegTree(max_depth=3).fit(X, y)
    sp = single.predict(X)
    single_mse = sum((y[i] - sp[i]) ** 2 for i in range(len(y))) / len(y)
    print(f"\n  A single depth-3 tree: MSE {single_mse:.4f} -- boosting is "
          f"{single_mse / gbr.mse(X, y):.0f}x better.")

    print("\n  Learning rate vs number of trees (shrinkage regularizes):")
    for lr in (0.02, 0.1, 0.5):
        for nt in (10, 40, 120):
            g = GradientBoostingRegressor(n_estimators=nt, learning_rate=lr, max_depth=3).fit(X, y)
            print(f"    lr={lr:<4} trees={nt:>3}: MSE {g.mse(X, y):.4f}")

    # classification on a circular boundary
    Xc, yc = [], []
    for _ in range(200):
        a, b = rng() * 4 - 2, rng() * 4 - 2
        yc.append(1 if a * a + b * b < 1.5 else 0)
        Xc.append([a, b])
    gbc = GradientBoostingClassifier(n_estimators=80, learning_rate=0.2, max_depth=3).fit(Xc, yc)
    print(f"\n  Classification on a circular boundary (a nonlinear problem for a linear model):")
    print(f"    accuracy {gbc.accuracy(Xc, yc):.1%}, log-loss {gbc.log_loss(Xc, yc):.3f}")

    print("\n  Boosting is gradient descent in function space: each tree fits the residual (the")
    print("  negative gradient of the loss), and a shrunken step of it is added to the model. Small")
    print("  learning rates plus many shallow trees is the recipe that wins tabular competitions.")

    _svg(os.path.join(outdir, "gradient_boosting.svg"), X, y, staged, losses, Xc, yc, gbc)
    print(f"\n  wrote {os.path.join(outdir, 'gradient_boosting.svg')}")


def _svg(path, X, y, staged, losses, Xc, yc, gbc, width=760, height=440):
    xs = [p[0] for p in X]
    xa, xb = min(xs), max(xs)
    ya, yb = min(y) - 0.4, max(y) + 0.4
    lx0, lx1 = 45, width // 2 - 15
    y0, y1 = height - 50, 70

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(v):
        return y0 - (v - ya) / (yb - ya) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Gradient boosting: the fit sharpens as trees accumulate</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'left: data with the ensemble fit after 1, 5, and 120 trees; right: training loss '
        f'falling</text>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.1"/>',
        f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.1"/>',
    ]
    # data points
    for i in range(len(X)):
        parts.append(f'<circle cx="{LX(xs[i]):.1f}" cy="{LY(y[i]):.1f}" r="2.2" '
                     f'fill="#8b949e" opacity="0.55"/>')
    # staged fits: light -> bright as more trees
    stages = [(0, "#3d3410"), (4, "#ffd43b"), (len(staged) - 1, "#06d6a0")]
    for idx, col in stages:
        if idx < len(staged):
            pred = staged[idx]
            pts = " ".join(f"{LX(xs[i]):.1f},{LY(pred[i]):.1f}" for i in range(len(X)))
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
    parts.append(f'<text x="{lx0+6}" y="{y1+4}" fill="#06d6a0" font-size="10">final (120 trees)</text>')
    parts.append(f'<text x="{lx0+6}" y="{y1+18}" fill="#ffd43b" font-size="10">5 trees</text>')
    parts.append(f'<text x="{lx0+6}" y="{y1+32}" fill="#3d3410" font-size="10">1 tree</text>')

    # right: loss curve (log scale)
    rx0, rx1 = width // 2 + 45, width - 30
    ry0, ry1 = y0, y1
    logs = [math.log10(max(l, 1e-6)) for l in losses]
    lo, hi = min(logs), max(logs)

    def RX(i):
        return rx0 + i / (len(losses) - 1) * (rx1 - rx0)

    def RY(v):
        return ry0 - (v - lo) / (hi - lo + 1e-9) * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.1"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.1"/>')
    lpts = " ".join(f"{RX(i):.1f},{RY(logs[i]):.1f}" for i in range(len(losses)))
    parts.append(f'<polyline points="{lpts}" fill="none" stroke="#ff6b6b" stroke-width="2.2"/>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">trees added -> training MSE (log scale)</text>')
    parts.append(f'<text x="{rx0-4:.1f}" y="{ry1:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">1e{int(hi)}</text>')
    parts.append(f'<text x="{rx0-4:.1f}" y="{ry0:.1f}" fill="#8b949e" font-size="9" '
                 f'text-anchor="end">1e{int(lo)}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

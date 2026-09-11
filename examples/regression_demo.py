"""Demo: linear and logistic regression.

Fits a line to noisy data (reporting R^2) and a logistic classifier to a 1D two-class problem
(reporting accuracy and the decision boundary), and shows the log-loss falling as training
proceeds. Draws the linear fit through the points and the logistic sigmoid with its boundary.

    python examples/regression_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from regression import (linear_fit, linear_fit_gd, predict, r_squared,  # noqa: E402
                        logistic_fit, predict_proba, accuracy, log_loss, sigmoid)


def _noisy_line(slope, intercept, n, noise, seed):
    state = seed
    pts = []
    for i in range(n):
        x = i * 8.0 / (n - 1)
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        u = (state >> 8) / (1 << 24)
        pts.append((x, slope * x + intercept + noise * (u - 0.5) * 2))
    return pts


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    pts = _noisy_line(1.5, 2.0, 30, 3.0, seed=7)
    X = [[p[0]] for p in pts]
    y = [p[1] for p in pts]
    w = linear_fit(X, y)
    wgd = linear_fit_gd(X, y, lr=0.01, epochs=8000)
    print("Linear regression: least-squares line through noisy data\n")
    print(f"  QR fit:   y = {w[1]:.3f} x + {w[0]:.3f}  (true slope 1.5, intercept 2.0)")
    print(f"  GD fit:   y = {wgd[1]:.3f} x + {wgd[0]:.3f}  (matches QR)")
    print(f"  R^2 = {r_squared(X, y, w):.4f}\n")

    # logistic: two classes along a line, boundary near x = 5
    Xc = [[x * 0.5] for x in range(21)]        # x = 0, 0.5, ..., 10
    yc = [1 if row[0] > 5 else 0 for row in Xc]
    wl = logistic_fit(Xc, yc, lr=0.8, epochs=6000)
    print("Logistic regression: probability of class 1 vs x (boundary where p = 0.5)")
    print(f"  accuracy {accuracy(Xc, yc, wl):.1%}, log-loss {log_loss(Xc, yc, wl):.4f}")
    print(f"  decision boundary at x = {-wl[0] / wl[1]:.2f}  (data splits at 5)\n")

    print("  Log-loss falls as gradient descent proceeds:")
    for e in (10, 100, 1000, 6000):
        print(f"    {e:>5} epochs: log-loss {log_loss(Xc, yc, logistic_fit(Xc, yc, lr=0.8, epochs=e)):.4f}")
    print("\n  Linear regression fits a line by minimizing squared error (closed form or GD);")
    print("  logistic squashes w.x+b through the sigmoid to a probability and minimizes log-loss,")
    print("  a convex objective with one global minimum. Both are the base of modern ML pipelines.")

    _svg(os.path.join(outdir, "regression.svg"), pts, w, Xc, yc, wl)
    print(f"\n  wrote {os.path.join(outdir, 'regression.svg')}")


def _svg(path, pts, w, Xc, yc, wl, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Regression: a fitted line (left) and a logistic decision boundary (right)</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'least-squares line through noisy points; sigmoid probability crossing 0.5 at the boundary</text>',
    ]

    # left: linear fit
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xa, xb = min(xs), max(xs)
    ya, yb = min(ys) - 1, max(ys) + 1
    lx0, lx1 = 45, width // 2 - 20
    y0, y1 = height - 45, 65

    def LX(x):
        return lx0 + (x - xa) / (xb - xa) * (lx1 - lx0)

    def LY(y):
        return y0 - (y - ya) / (yb - ya) * (y0 - y1)

    parts.append(f'<line x1="{lx0}" y1="{y0}" x2="{lx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{y0}" x2="{lx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    for px, py in pts:
        parts.append(f'<circle cx="{LX(px):.1f}" cy="{LY(py):.1f}" r="3" fill="#4dabf7" opacity="0.7"/>')
    parts.append(f'<line x1="{LX(xa):.1f}" y1="{LY(w[1]*xa+w[0]):.1f}" '
                 f'x2="{LX(xb):.1f}" y2="{LY(w[1]*xb+w[0]):.1f}" stroke="#06d6a0" stroke-width="2.5"/>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">linear least squares</text>')

    # right: logistic sigmoid + points + boundary
    rx0, rx1 = width // 2 + 45, width - 30
    xmax = Xc[-1][0]

    def RX(x):
        return rx0 + x / xmax * (rx1 - rx0)

    def RY(p):
        return y0 - p * (y0 - y1)

    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{y0}" x2="{rx0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{RY(0.5):.1f}" x2="{rx1}" y2="{RY(0.5):.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    # sigmoid curve
    curve = []
    steps = 200
    for i in range(steps + 1):
        x = xmax * i / steps
        p = sigmoid(wl[1] * x + wl[0])
        curve.append(f"{RX(x):.1f},{RY(p):.1f}")
    parts.append(f'<polyline points="{" ".join(curve)}" fill="none" stroke="#ff922b" stroke-width="2.5"/>')
    # class points at p=0 / p=1
    for row, label in zip(Xc, yc):
        parts.append(f'<circle cx="{RX(row[0]):.1f}" cy="{RY(label):.1f}" r="3" '
                     f'fill="{"#06d6a0" if label else "#ff6b6b"}" opacity="0.7"/>')
    # decision boundary
    bx = -wl[0] / wl[1]
    parts.append(f'<line x1="{RX(bx):.1f}" y1="{y1}" x2="{RX(bx):.1f}" y2="{y0}" '
                 f'stroke="#ffd43b" stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{RX(bx):.1f}" y="{y1-4:.1f}" fill="#ffd43b" font-size="9" '
                 f'text-anchor="middle">boundary x={bx:.1f}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">logistic: P(class 1) vs x</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

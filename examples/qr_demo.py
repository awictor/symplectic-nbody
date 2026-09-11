"""Demo: QR decomposition and least-squares fitting.

Factors a matrix into Q R (orthonormal times upper-triangular), then uses it to fit a line and
a quadratic to scattered data by least squares. Draws the noisy points with the fitted line and
the residuals, and confirms Q's columns are orthonormal.

    python examples/qr_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qr import (qr_decompose, lstsq, is_orthonormal, reconstruct,  # noqa: E402
                residual_norm, matvec)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
    Q, R = qr_decompose(A)
    print("QR decomposition: A = Q R, Q orthonormal, R upper-triangular\n")
    print(f"  Q^T Q = I (orthonormal columns): {is_orthonormal(Q)}")
    print(f"  Q R reconstructs A: {all(abs(reconstruct(Q, R)[i][j] - A[i][j]) < 1e-9 for i in range(3) for j in range(3))}")
    print("  R (upper-triangular):", [[round(v, 2) for v in row] for row in R], "\n")

    # least-squares line fit to noisy data
    pts = _noisy_line(slope=1.7, intercept=3.0, n=25, noise=2.5, seed=7)
    Am = [[px, 1] for px, _ in pts]
    bm = [py for _, py in pts]
    slope, intercept = lstsq(Am, bm)
    print(f"  Least-squares line fit to {len(pts)} noisy points (true slope 1.7, intercept 3.0):")
    print(f"    fitted y = {slope:.3f} x + {intercept:.3f}")
    print(f"    residual ||Ax - b|| = {residual_norm(Am, [slope, intercept], bm):.3f}")

    # quadratic fit, exact data
    qpts = [(px, 3 * px * px - 2 * px + 5) for px in range(8)]
    Aq = [[px * px, px, 1] for px, _ in qpts]
    bq = [py for _, py in qpts]
    a, b, c = lstsq(Aq, bq)
    print(f"\n  Least-squares quadratic fit (exact data): {a:.2f} x^2 + {b:.2f} x + {c:.2f}  (true 3, -2, 5)")

    print("\n  QR solves least squares via R x = Q^T b -- more stable than the normal equations")
    print("  A^T A x = A^T b (which squares the condition number). It also powers eigenvalue")
    print("  iteration and orthogonal regression. Modified Gram-Schmidt keeps Q orthogonal.")

    _svg(os.path.join(outdir, "qr.svg"), pts, slope, intercept)
    print(f"\n  wrote {os.path.join(outdir, 'qr.svg')}")


def _noisy_line(slope, intercept, n, noise, seed):
    state = seed
    pts = []
    for i in range(n):
        x = i * 10.0 / (n - 1)
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        u = (state >> 8) / (1 << 24)
        pts.append((x, slope * x + intercept + noise * (u - 0.5) * 2))
    return pts


def _svg(path, pts, slope, intercept, w=760, h=400):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xa, xb = min(xs), max(xs)
    ya, yb = min(ys), max(ys)
    pad = 0.1 * (yb - ya)
    ya -= pad
    yb += pad

    x0, x1 = 55, w - 30
    y0, y1 = h - 55, 65

    def X(x):
        return x0 + (x - xa) / (xb - xa) * (x1 - x0)

    def Y(y):
        return y0 - (y - ya) / (yb - ya) * (y0 - y1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'QR least squares: the best-fit line through scattered data</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'residuals (gray) minimized in the sum-of-squares sense; the fit line is orthogonal to them</text>',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>',
    ]

    # residual segments from each point to the line
    for px, py in pts:
        fy = slope * px + intercept
        parts.append(f'<line x1="{X(px):.1f}" y1="{Y(py):.1f}" x2="{X(px):.1f}" y2="{Y(fy):.1f}" '
                     f'stroke="#8b949e" stroke-width="1" opacity="0.5"/>')
    # fit line
    parts.append(f'<line x1="{X(xa):.1f}" y1="{Y(slope*xa+intercept):.1f}" '
                 f'x2="{X(xb):.1f}" y2="{Y(slope*xb+intercept):.1f}" '
                 f'stroke="#06d6a0" stroke-width="2.5"/>')
    # points
    for px, py in pts:
        parts.append(f'<circle cx="{X(px):.1f}" cy="{Y(py):.1f}" r="3" fill="#4dabf7"/>')
    parts.append(f'<text x="{X(xb):.1f}" y="{Y(slope*xb+intercept)-8:.1f}" fill="#06d6a0" '
                 f'font-size="10" text-anchor="end">y = {slope:.2f}x + {intercept:.2f}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">x -> y</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

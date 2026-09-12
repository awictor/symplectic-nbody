"""Demo: Householder QR -- orthogonal factorization by reflections.

Factorizes a matrix into Q R by Householder reflections, verifies orthonormality and reconstruction,
and fits a least-squares line to noisy data. Draws the data with the fitted line.

    python examples/householder_qr_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from householder_qr import (qr, lstsq, reconstruct, is_orthonormal,  # noqa: E402
                            is_upper_triangular)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Householder QR: A = Q R by orthogonal reflections\n")

    A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
    Q, R = qr(A)
    print("  A =")
    for row in A:
        print("    " + "  ".join(f"{v:6.1f}" for v in row))
    print("\n  R (upper triangular) =")
    for row in R:
        print("    " + "  ".join(f"{v:8.3f}" for v in row))
    print(f"\n  Q orthonormal (Q^T Q = I): {is_orthonormal(Q)}")
    print(f"  R upper triangular: {is_upper_triangular(R)}")
    recon = reconstruct(Q, R)
    max_err = max(abs(recon[i][j] - A[i][j]) for i in range(3) for j in range(3))
    print(f"  reconstruction error |QR - A|: {max_err:.2e}")

    # least-squares line fit y = a + b x to noisy data
    xs = [0, 1, 2, 3, 4, 5, 6, 7]
    ys = [1.1, 2.9, 5.2, 6.8, 9.1, 10.8, 13.2, 14.9]     # roughly y = 1 + 2x
    design = [[1.0, x] for x in xs]
    a, b = lstsq(design, ys)
    print(f"\n  least-squares line fit to 8 noisy points: y = {a:.3f} + {b:.3f} x")
    print(f"  (data generated near y = 1 + 2x)")

    print("\n  Each reflection H = I - 2 v v^T / (v^T v) mirrors a column onto the axis, zeroing every")
    print("  entry below the diagonal in one shot; n reflections triangularize A into R, and their")
    print("  product is Q. Because reflections are EXACTLY orthogonal, Q stays orthonormal to machine")
    print("  precision even on ill-conditioned inputs where Gram-Schmidt drifts.")

    _svg(os.path.join(outdir, "householder_qr.svg"), xs, ys, a, b)
    print(f"\n  wrote {os.path.join(outdir, 'householder_qr.svg')}")


def _svg(path, xs, ys, a, b, width=680, height=420):
    ox, oy = 60, 360
    pw, ph = width - 100, 300
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ymin = min(ymin, a)
    ymax = max(ymax, a + b * xmax)

    def px(x):
        return ox + (x - xmin) / (xmax - xmin) * pw

    def py(y):
        return oy - (y - ymin) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'Least-squares line fit via Householder QR</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'y = {a:.2f} + {b:.2f} x, the line minimising squared vertical error to the points</text>',
    ]
    # axes
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d"/>')
    # fitted line
    parts.append(f'<line x1="{px(xmin):.1f}" y1="{py(a + b*xmin):.1f}" '
                 f'x2="{px(xmax):.1f}" y2="{py(a + b*xmax):.1f}" stroke="#06d6a0" '
                 f'stroke-width="2.5"/>')
    # residual segments + points
    for x, y in zip(xs, ys):
        yp = a + b * x
        parts.append(f'<line x1="{px(x):.1f}" y1="{py(y):.1f}" x2="{px(x):.1f}" '
                     f'y2="{py(yp):.1f}" stroke="#ff922b" stroke-width="1" opacity="0.6"/>')
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" fill="#4dabf7"/>')

    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+30}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">orange stubs are residuals the fit minimises</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

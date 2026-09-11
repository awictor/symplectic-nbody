"""Demo: conjugate gradient -- solving big SPD systems with mat-vecs only.

Solves a symmetric positive-definite system by CG (checked against LU), shows the residual
plunging to machine precision within n steps, and contrasts CG's fast convergence with the
zig-zag of plain steepest descent. Draws the residual-decay curves.

    python examples/conjugate_gradient_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conjugate_gradient import (conjugate_gradient, preconditioned_cg,  # noqa: E402
                                make_spd, spd_reference, residual_norm, energy,
                                _matvec, _dot, _axpy)


def steepest_descent(A, b, tol=1e-10, max_iter=500):
    """Plain steepest descent (gradient descent on the energy) -- for contrast with CG."""
    n = len(A)
    x = [0.0] * n
    r = [b[i] - _matvec(A, x)[i] for i in range(n)]
    bnorm = math.sqrt(_dot(b, b)) or 1.0
    hist = [math.sqrt(_dot(r, r))]
    for _ in range(max_iter):
        if math.sqrt(_dot(r, r)) <= tol * bnorm:
            break
        Ar = _matvec(A, r)
        alpha = _dot(r, r) / _dot(r, Ar)
        x = _axpy(alpha, r, x)
        r = [b[i] - _matvec(A, x)[i] for i in range(n)]
        hist.append(math.sqrt(_dot(r, r)))
    return x, hist


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    A = make_spd(20, seed=3)
    b = [float(i + 1) for i in range(20)]
    x, hist = conjugate_gradient(A, b)
    print("Conjugate gradient: minimize (1/2)x^T A x - b^T x with mat-vecs only\n")
    print(f"  20x20 SPD system:")
    print(f"    CG converged in {len(hist) - 1} iterations (<= n = 20)")
    print(f"    residual ||A x - b|| = {residual_norm(A, x, b):.2e}")
    print(f"    matches LU: {all(abs(a - c) < 1e-6 for a, c in zip(x, spd_reference(A, b)))}\n")

    sd_x, sd_hist = steepest_descent(A, b)
    print(f"  Steepest descent on the same system: {len(sd_hist) - 1} iterations to the same tol")
    print(f"    -- CG picks A-conjugate directions so it never undoes earlier progress;")
    print(f"    steepest descent zig-zags: {(len(sd_hist)-1) / (len(hist)-1):.1f}x as many steps.\n")

    px, ph = preconditioned_cg(A, b)
    print(f"  Jacobi-preconditioned CG: {len(ph) - 1} iterations "
          f"(residual {residual_norm(A, px, b):.1e})")
    print("\n  No factorization, O(nnz) per step, O(n) memory: CG (and its preconditioned")
    print("  cousins) solve the million-unknown SPD systems of finite elements, image")
    print("  reconstruction, and graph Laplacians that a dense LU could never store.")

    _svg(os.path.join(outdir, "conjugate_gradient.svg"), hist, sd_hist)
    print(f"\n  wrote {os.path.join(outdir, 'conjugate_gradient.svg')}")


def _svg(path, cg_hist, sd_hist, w=760, h=390):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Conjugate gradient converges in <= n steps; steepest descent crawls</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'residual norm vs iteration (log scale): CG plunges, steepest descent zig-zags slowly</text>',
    ]

    x0, x1 = 60, w - 40
    y0, y1 = h - 55, 62
    imax = max(len(cg_hist), len(sd_hist)) - 1
    r0 = max(cg_hist[0], sd_hist[0])
    emin, emax = -12.0, math.log10(r0) + 0.2

    def X(i):
        return x0 + i / imax * (x1 - x0)

    def Y(r):
        le = max(emin, min(emax, math.log10(max(r, 1e-13))))
        return y0 - (le - emin) / (emax - emin) * (y0 - y1)

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    for e in range(-12, int(emax) + 1, 3):
        yy = Y(10.0 ** e)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-6:.1f}" y="{yy+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">1e{e}</text>')
    for hist, col, lab in ((sd_hist, "#ff6b6b", "steepest descent"),
                           (cg_hist, "#06d6a0", "conjugate gradient")):
        pts = " ".join(f"{X(i):.1f},{Y(r):.1f}" for i, r in enumerate(hist))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
    # legend
    parts.append(f'<rect x="{x1-180}" y="{y1}" width="10" height="10" fill="#06d6a0"/>'
                 f'<text x="{x1-165}" y="{y1+9}" fill="#e6edf3" font-size="9">conjugate gradient</text>')
    parts.append(f'<rect x="{x1-180}" y="{y1+15}" width="10" height="10" fill="#ff6b6b"/>'
                 f'<text x="{x1-165}" y="{y1+24}" fill="#e6edf3" font-size="9">steepest descent</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">iteration -> residual norm</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

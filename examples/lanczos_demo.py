"""Demo: the Lanczos algorithm -- extreme eigenvalues of a huge matrix you never store.

Finds the top eigenvalues of a random symmetric matrix (matching a dense solver in a fraction of the
iterations), then runs matrix-free on a large sparse graph Laplacian to recover its spectral gap.
Draws convergence of the largest Ritz value vs iteration and the Laplacian spectrum.

    python examples/lanczos_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lanczos import (eigenvalues, largest_eigenvalues, smallest_eigenvalues,  # noqa: E402
                     matvec_from_matrix)
import jacobi_eigen  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24) * 2 - 1


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Lanczos: extreme eigenvalues via matrix-vector products, no full matrix needed\n")

    rng = LCG(2024)
    n = 40
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = rng.u() * 3
            A[i][j] = v
            A[j][i] = v
    jv = sorted(jacobi_eigen.sorted_eigen(A)[0])
    mv = matvec_from_matrix(A)

    print(f"  random symmetric {n}x{n} matrix:")
    print(f"    true largest eigenvalue  (dense): {jv[-1]:.6f}")
    print(f"    true smallest eigenvalue (dense): {jv[0]:.6f}")
    print("    Lanczos largest, by iteration count m:")
    conv = []
    for m in (4, 8, 12, 20, 40):
        est = largest_eigenvalues(mv, n, k=1, m=m)[0]
        conv.append((m, est))
        print(f"      m={m:2d}: {est:.6f}  (error {abs(est-jv[-1]):.2e})")
    print(f"    -> converged to full accuracy in far fewer than {n} steps\n")

    # matrix-free sparse graph Laplacian: a ring of n nodes
    N = 500
    def ring_laplacian_mv(x):
        return [2 * x[i] - x[(i - 1) % N] - x[(i + 1) % N] for i in range(N)]

    print(f"  matrix-free: Laplacian of a {N}-node ring graph (never stored as a matrix):")
    big = largest_eigenvalues(ring_laplacian_mv, N, k=1, m=60)
    small = smallest_eigenvalues(ring_laplacian_mv, N, k=1, m=150)
    print(f"    largest eigenvalue:  {big[0]:.5f}  (analytic max 4 -- outermost, converges fast)")
    print(f"    smallest eigenvalue: {small[0]:.2e}  (analytic 0, the constant null vector)")
    print("    (the tightly clustered small eigenvalues need more iterations than the extremes)\n")

    print("  The Krylov vectors v, Av, A^2 v, ... quickly span the dominant eigen-directions; a")
    print("  three-term recurrence orthogonalises them into a tiny tridiagonal matrix whose eigenvalues")
    print("  (Ritz values) approximate the extremes. An n-million eigenproblem becomes an m-by-m one.")

    _svg(os.path.join(outdir, "lanczos.svg"), conv, jv[-1], eigenvalues(ring_laplacian_mv, N, m=80))
    print(f"\n  wrote {os.path.join(outdir, 'lanczos.svg')}")


def _svg(path, conv, true_max, laplacian_spectrum, width=760, height=440):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'Lanczos: largest Ritz value converging (top), ring Laplacian spectrum (bottom)</text>',
    ]

    # top: convergence
    ox, oy = 60, 210
    pw, ph = width - 100, 150
    ms = [m for m, _ in conv]
    errs = [max(abs(e - true_max), 1e-16) for _, e in conv]
    mmax = max(ms)
    # log-scale the error
    logerrs = [math.log10(e) for e in errs]
    lo, hi = min(logerrs), max(logerrs)
    span = hi - lo or 1

    def px(m):
        return ox + m / mmax * pw

    def py(le):
        return oy - (le - lo) / span * ph

    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d" stroke-width="1"/>')
    pts = " ".join(f"{px(m):.1f},{py(le):.1f}" for m, le in zip(ms, logerrs))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2"/>')
    for m, le in zip(ms, logerrs):
        parts.append(f'<circle cx="{px(m):.1f}" cy="{py(le):.1f}" r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+26}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">Lanczos iterations m</text>')
    parts.append(f'<text x="20" y="{oy-ph-2:.0f}" fill="#8b949e" font-size="11">log10 error in largest eigenvalue</text>')

    # bottom: laplacian spectrum as sorted dots
    bx, by = 60, 410
    bw, bh = width - 100, 130
    spec = sorted(laplacian_spectrum)
    smin, smax = spec[0], spec[-1]
    sp = smax - smin or 1
    parts.append(f'<text x="{bx}" y="{by-bh-6}" fill="#8b949e" font-size="11">'
                 f'ring Laplacian eigenvalues (0 to 4), forming a cosine band</text>')
    for i, v in enumerate(spec):
        x = bx + i / (len(spec) - 1) * bw
        y = by - (v - smin) / sp * bh
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.6" fill="#06d6a0"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

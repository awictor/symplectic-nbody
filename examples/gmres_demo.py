"""Demo: GMRES -- solving a nonsymmetric linear system by minimising the residual in a Krylov subspace.

Solves a nonsymmetric convection-diffusion-style system matrix-free, shows the residual dropping
monotonically, demonstrates that a Jacobi preconditioner accelerates an ill-scaled system, and draws
the residual-vs-iteration convergence curves.

    python examples/gmres_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gmres import gmres, gmres_restart, matvec_from_matrix, true_residual_norm  # noqa: E402


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24) * 2 - 1


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("GMRES: minimal-residual Krylov solver for nonsymmetric systems\n")

    # matrix-free 1D convection-diffusion: -u'' + c u' discretised, nonsymmetric from the drift
    N = 80
    c = 3.0
    def op(x):
        out = [0.0] * N
        for i in range(N):
            out[i] = 2 * x[i]
            if i > 0:
                out[i] += (-1 - c) * x[i - 1]
            if i < N - 1:
                out[i] += (-1 + c) * x[i + 1]
        return out
    b = [1.0] * N

    x, res, conv = gmres(op, b, tol=1e-10)
    print(f"  {N}-point convection-diffusion operator (nonsymmetric, matrix-free):")
    print(f"    converged: {conv} in {len(res)-1} iterations")
    print(f"    true residual ||b - A x||: {true_residual_norm(op, b, x):.2e}")
    print(f"    residual monotonically non-increasing: "
          f"{all(res[i] >= res[i+1]-1e-12 for i in range(len(res)-1))}\n")

    # ill-scaled system: preconditioner helps
    rng = LCG(2024)
    n = 40
    A = [[rng.u() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        A[i][i] += n
    scales = [10 ** (rng.u() * 3) for _ in range(n)]
    A = [[A[i][j] * scales[i] for j in range(n)] for i in range(n)]
    rhs = [rng.u() for _ in range(n)]
    mv = matvec_from_matrix(A)
    diag = [A[i][i] for i in range(n)]

    _, res_none, _ = gmres(mv, rhs, tol=1e-8, max_iter=n)
    _, res_prec, _ = gmres(mv, rhs, tol=1e-8, max_iter=n,
                           precond=lambda r: [r[i] / diag[i] for i in range(n)])
    print("  ill-scaled system (diagonal spanning 3 orders of magnitude):")
    print(f"    without preconditioner: {len(res_none)-1} iterations")
    print(f"    with Jacobi precondition: {len(res_prec)-1} iterations "
          f"({(len(res_none)-1)/(max(1,len(res_prec)-1)):.1f}x fewer)\n")

    print("  GMRES builds an orthonormal Krylov basis by Arnoldi iteration and, at every step, returns")
    print("  the subspace vector that MINIMISES ||b - A x|| via a small Givens-rotated least-squares")
    print("  solve -- so the residual can only fall. Unlike conjugate gradient it needs no symmetry,")
    print("  which is why it is the default solver for convection, electromagnetics, and CFD.")

    _svg(os.path.join(outdir, "gmres.svg"), res, res_none, res_prec)
    print(f"\n  wrote {os.path.join(outdir, 'gmres.svg')}")


def _svg(path, res_main, res_none, res_prec, width=760, height=400):
    ox, oy = 60, 330
    pw, ph = width - 100, 260

    def curve_pts(res, colour):
        n = len(res)
        logs = [math.log10(max(r, 1e-16)) for r in res]
        lo, hi = -12, 0.5
        def px(i):
            return ox + i / max(1, maxlen - 1) * pw
        def py(l):
            return oy - (l - lo) / (hi - lo) * ph
        pts = " ".join(f"{px(i):.1f},{py(logs[i]):.1f}" for i in range(n))
        return f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>'

    maxlen = max(len(res_main), len(res_none), len(res_prec))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="17">'
        f'GMRES residual convergence (log scale) -- always decreasing</text>',
    ]
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-ph}" stroke="#30363d" stroke-width="1"/>')
    # grid lines at 10^-4, 10^-8
    for lval, lab in ((0, "1"), (-4, "1e-4"), (-8, "1e-8"), (-12, "1e-12")):
        y = oy - (lval - (-12)) / 12.5 * ph
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+pw}" y2="{y:.1f}" stroke="#21262d" '
                     f'stroke-width="0.6"/>')
        parts.append(f'<text x="{ox-6}" y="{y+4:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{lab}</text>')

    parts.append(curve_pts(res_main, "#4dabf7"))
    parts.append(curve_pts(res_none, "#ff922b"))
    parts.append(curve_pts(res_prec, "#06d6a0"))
    parts.append(f'<text x="{ox+pw-220}" y="{oy-ph+18}" fill="#4dabf7" font-size="11">convection-diffusion</text>')
    parts.append(f'<text x="{ox+pw-220}" y="{oy-ph+36}" fill="#ff922b" font-size="11">ill-scaled, no precond</text>')
    parts.append(f'<text x="{ox+pw-220}" y="{oy-ph+54}" fill="#06d6a0" font-size="11">ill-scaled + Jacobi precond</text>')
    parts.append(f'<text x="{ox+pw/2:.0f}" y="{oy+28}" fill="#8b949e" font-size="12" '
                 f'text-anchor="middle">iteration</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

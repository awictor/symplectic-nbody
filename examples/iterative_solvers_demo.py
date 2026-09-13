"""Demo: stationary iterative solvers on the 2-D Poisson equation.

Solves -Laplacian(u) = f on the unit square with Jacobi, Gauss-Seidel, and optimal SOR, compares
iteration counts (showing SOR's dramatic acceleration), verifies the solution against a manufactured
analytic answer, and draws the solution field as a heatmap.

    python examples/iterative_solvers_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from iterative_solvers import poisson_2d, optimal_sor_omega  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Stationary iterative solvers: Jacobi, Gauss-Seidel, SOR on the 2-D Poisson equation\n")

    def f(x, y):
        return 2 * math.pi ** 2 * math.sin(math.pi * x) * math.sin(math.pi * y)

    def u_exact(x, y):
        return math.sin(math.pi * x) * math.sin(math.pi * y)

    n = 14
    h = 1.0 / (n + 1)
    print(f"  -Laplacian(u) = 2 pi^2 sin(pi x) sin(pi y) on a {n}x{n} interior grid.")
    print(f"  Manufactured exact solution u = sin(pi x) sin(pi y).\n")

    print(f"  {'method':>14}  {'iterations':>11}  {'max error':>10}")
    results = {}
    for method in ["jacobi", "gauss_seidel", "sor"]:
        grid, iters = poisson_2d(f, n, method=method, tol=1e-8, max_iter=100000)
        maxerr = max(abs(grid[i][j] - u_exact((j + 1) * h, (i + 1) * h))
                     for i in range(n) for j in range(n))
        results[method] = (grid, iters, maxerr)
        print(f"  {method:>14}  {iters:>11}  {maxerr:>10.2e}")

    it_j = results["jacobi"][1]
    it_s = results["sor"][1]
    print(f"\n  Optimal SOR omega = {optimal_sor_omega(n):.4f}")
    print(f"  SOR is {it_j / it_s:.1f}x fewer iterations than Jacobi -- O(N) vs O(N^2) scaling.")
    print("  All three converge to the same solution; only the speed differs.")

    _svg(os.path.join(outdir, "iterative_solvers.svg"), results["sor"][0], n,
         [(m, results[m][1]) for m in ["jacobi", "gauss_seidel", "sor"]])
    print(f"\n  wrote {os.path.join(outdir, 'iterative_solvers.svg')}")


def _svg(path, grid, n, iters, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Left: Poisson solution field (SOR). Right: iteration counts by method</text>',
    ]

    # ---- left: heatmap ------------------------------------------------------------------
    ox, oy = 40, 55
    cell = min(300 // n, 14)
    vmax = max(max(row) for row in grid)
    for i in range(n):
        for j in range(n):
            t = grid[i][j] / vmax if vmax > 0 else 0
            r = int(0x0d + t * (0xff - 0x0d))
            g = int(0x11 + t * (0xd4 - 0x11))
            b = int(0x17 + t * (0x3b - 0x17))
            parts.append(f'<rect x="{ox + j*cell}" y="{oy + i*cell}" width="{cell}" '
                         f'height="{cell}" fill="rgb({r},{g},{b})"/>')
    parts.append(f'<text x="{ox}" y="{oy + n*cell + 18}" fill="#8b949e" font-size="10">'
                 f'u(x,y) = sin(pi x) sin(pi y), peak at centre</text>')

    # ---- right: iteration bar chart (log scale) ----------------------------------------
    bx, by, bw, bh = 440, 70, 260, 260
    cols = {"jacobi": "#ff6b6b", "gauss_seidel": "#ffd43b", "sor": "#06d6a0"}
    maxit = max(it for _, it in iters)
    logmax = math.log10(maxit)
    barw = bw / len(iters)
    parts.append(f'<line x1="{bx}" y1="{by+bh}" x2="{bx+bw}" y2="{by+bh}" stroke="#8b949e"/>')
    for idx, (m, it) in enumerate(iters):
        h = bh * math.log10(it) / logmax
        x = bx + idx * barw
        parts.append(f'<rect x="{x+6:.0f}" y="{by+bh-h:.0f}" width="{barw-12:.0f}" '
                     f'height="{h:.0f}" fill="{cols[m]}"/>')
        parts.append(f'<text x="{x+barw/2:.0f}" y="{by+bh-h-6:.0f}" fill="#e6edf3" '
                     f'font-size="10" text-anchor="middle">{it}</text>')
        parts.append(f'<text x="{x+barw/2:.0f}" y="{by+bh+16:.0f}" fill="{cols[m]}" '
                     f'font-size="8" text-anchor="middle">{m[:8]}</text>')
    parts.append(f'<text x="{bx}" y="{by-6}" fill="#8b949e" font-size="9">'
                 f'iterations to converge (log scale)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

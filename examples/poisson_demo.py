"""Demo: solving the Poisson/Laplace equation by relaxation, and comparing the methods.

Solves Laplace on a plate with hot/cold edges (steady-state temperature), solves Poisson for a point
charge's potential, and compares Jacobi/Gauss-Seidel/SOR convergence speed. Draws the temperature
field as a heatmap.

    python examples/poisson_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poisson import solve, mean_value_error, residual  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Poisson/Laplace by relaxation: steady-state fields on a grid\n")

    # a heated plate: hot left edge, cold right edge, insulated-ish top/bottom ramp
    N = 40

    def plate_boundary(i, j):
        if i == 0:
            return 100.0            # hot left
        if i == N - 1:
            return 0.0              # cold right
        # top and bottom: linear ramp between them
        return 100.0 * (1 - i / (N - 1))

    u, sweeps = solve(N, N, boundary=plate_boundary, method="sor", tol=1e-8)
    print(f"  Steady-state heat on a {N}x{N} plate (hot left 100, cold right 0):")
    print(f"    converged in {sweeps} SOR sweeps; harmonic (mean-value error {mean_value_error(u):.2e})")
    print(f"    center temperature: {u[N//2][N//2]:.1f} (halfway between the edges)")

    # method comparison
    print("\n  Convergence speed (sweeps to tol=1e-8, {0}x{0} grid):".format(N))
    for method in ["jacobi", "gauss_seidel", "sor"]:
        _, sw = solve(N, N, boundary=plate_boundary, method=method, tol=1e-8)
        print(f"    {method:14}: {sw:5d} sweeps")
    print("    SOR over-relaxes each correction by omega in (1,2), converging an order faster.")

    # Poisson: a point charge
    mid = N // 2
    def charge(i, j):
        return -50.0 if (i, j) == (mid, mid) else 0.0
    uc, _ = solve(N, N, boundary=lambda i, j: 0.0, source=charge, method="sor", tol=1e-8)
    print(f"\n  Poisson for a point charge (grounded box): peak potential {uc[mid][mid]:.2f}")
    print(f"    residual |laplacian(u) - f|: {residual(uc, source=charge):.2e}")

    print("\n  Laplace's equation makes every interior point the average of its neighbours -- a")
    print("  harmonic field, smooth with no interior hot spots. Relaxation just sweeps the grid")
    print("  averaging until it settles; SOR accelerates by overshooting each averaging step.")

    _svg(os.path.join(outdir, "poisson.svg"), u, N)
    print(f"\n  wrote {os.path.join(outdir, 'poisson.svg')}")


def _svg(path, u, N, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Laplace: steady-state temperature on a plate (hot left, cold right)</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'each interior cell is the average of its neighbours; isotherms curve smoothly across the plate</text>',
    ]

    # heatmap
    cell = min((width - 300) / N, (height - 100) / N)
    ox, oy = 40, 70
    vmin = min(min(row) for row in u)
    vmax = max(max(row) for row in u)
    vr = vmax - vmin or 1
    for i in range(N):
        for j in range(N):
            t = (u[i][j] - vmin) / vr
            # blue (cold) -> red (hot) through white
            if t < 0.5:
                r, g, b = int(40 + t * 2 * 200), int(60 + t * 2 * 180), int(180 + t * 2 * 40)
            else:
                s = (t - 0.5) * 2
                r, g, b = int(240), int(240 - s * 180), int(220 - s * 200)
            x = ox + i * cell
            y = oy + j * cell
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell+0.6:.1f}" '
                         f'height="{cell+0.6:.1f}" fill="rgb({min(r,255)},{min(g,255)},{min(b,255)})"/>')

    # contour lines (isotherms) at a few levels
    for level_frac in [0.25, 0.5, 0.75]:
        level = vmin + level_frac * vr
        for i in range(N - 1):
            for j in range(N - 1):
                # simple marching: if the level crosses between adjacent cells, dot it
                if (u[i][j] - level) * (u[i + 1][j] - level) < 0:
                    x = ox + (i + 0.5) * cell
                    y = oy + j * cell
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="0.7" fill="#0d1117"/>')

    # colour legend text
    parts.append(f'<text x="{ox + N*cell + 20:.0f}" y="90" fill="#ff6b6b" font-size="12">100 (hot)</text>')
    parts.append(f'<text x="{ox + N*cell + 20:.0f}" y="{oy + N*cell:.0f}" fill="#4dabf7" font-size="12">0 (cold)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

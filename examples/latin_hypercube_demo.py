"""Demo: Latin hypercube sampling vs plain Monte Carlo.

Compares the space-filling of a 2-D LHS design against random points, and shows LHS integration has
far lower error variance than Monte Carlo at the same sample count. Draws both point sets with the
stratification grid.

    python examples/latin_hypercube_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from latin_hypercube import (  # noqa: E402
    latin_hypercube,
    integrate,
    monte_carlo_integrate,
    estimator_variance,
    stratification_ok,
    min_pairwise_distance,
)


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Latin hypercube sampling: even coverage that beats random Monte Carlo\n")

    lhs = latin_hypercube(16, 2, seed=7)
    print(f"  16-point 2-D LHS design: valid stratification = {stratification_ok(lhs)}")
    print(f"  minimum inter-point distance {min_pairwise_distance(lhs):.4f} "
          f"(spread out, no clumps)\n")

    # variance comparison on an additive test function
    def f(x):
        return sum(math.sin(math.pi * xi) for xi in x)
    dim = 4
    true = dim * 2 / math.pi
    print(f"  Integrating sum sin(pi x_i) over [0,1]^{dim} (true value {true:.4f}):\n")
    print(f"  {'n':>5}  {'LHS MSE':>12}  {'Monte Carlo MSE':>16}  {'variance ratio':>14}")
    for n in [20, 40, 80, 160]:
        mse_lhs = estimator_variance(integrate, f, dim, n, true, trials=80)
        mse_mc = estimator_variance(monte_carlo_integrate, f, dim, n, true, trials=80)
        print(f"  {n:>5}  {mse_lhs:>12.3e}  {mse_mc:>16.3e}  {mse_mc/mse_lhs:>13.1f}x")

    print("\n  LHS forces every variable's range to be covered uniformly, so for near-additive")
    print("  functions its error variance is many times smaller than random sampling at equal cost.")

    # random points for the picture
    rng = _lcg(99)
    mc = [[rng(), rng()] for _ in range(16)]
    _svg(os.path.join(outdir, "latin_hypercube.svg"), lhs, mc, 16)
    print(f"\n  wrote {os.path.join(outdir, 'latin_hypercube.svg')}")


def _svg(path, lhs, mc, n, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Left: Latin hypercube (one per row & column). Right: random Monte Carlo (clumps and gaps)</text>',
    ]

    def panel(points, x0, title, col, grid):
        size = 300
        y0 = 50
        parts.append(f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="#161b22" '
                     f'stroke="#30363d"/>')
        if grid:
            for k in range(1, n):
                gx = x0 + size * k / n
                gy = y0 + size * k / n
                parts.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0+size}" '
                             f'stroke="#21262d" stroke-width="0.5"/>')
                parts.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0+size}" y2="{gy:.1f}" '
                             f'stroke="#21262d" stroke-width="0.5"/>')
        for p in points:
            parts.append(f'<circle cx="{x0 + p[0]*size:.1f}" cy="{y0 + p[1]*size:.1f}" r="4" '
                         f'fill="{col}"/>')
        parts.append(f'<text x="{x0 + size/2:.0f}" y="{y0+size+22:.0f}" fill="{col}" '
                     f'font-size="11" text-anchor="middle">{title}</text>')

    panel(lhs, 60, "Latin hypercube (stratified)", "#06d6a0", True)
    panel(mc, 420, "random Monte Carlo", "#ff6b6b", False)
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: Lasso regression -- automatic feature selection by L1 penalty.

Fits data generated from a few relevant features buried among many irrelevant ones, showing Lasso
zeros the noise features (unlike ridge), and traces the coefficient path as the L1 penalty grows.
Draws the coefficient paths.

    python examples/lasso_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lasso import lasso, ridge, lasso_path, n_nonzero, mse  # noqa: E402


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

    print("Lasso regression: L1 penalty selects a sparse subset of features\n")

    rng = _lcg(2024)
    p = 12
    true_coef = [0.0] * p
    true_coef[1] = 4.0
    true_coef[4] = -3.0
    true_coef[9] = 2.0
    n = 100
    X = [[rng() * 2 - 1 for _ in range(p)] for _ in range(n)]
    y = [sum(X[i][j] * true_coef[j] for j in range(p)) + 5 + (rng() - 0.5) * 0.4 for i in range(n)]

    true_active = [j for j in range(p) if true_coef[j] != 0]
    print(f"  {p} features, only {len(true_active)} relevant (indices {true_active}), {n} samples.\n")

    intercept, coef = lasso(X, y, lam=0.05)
    _, rcoef = ridge(X, y, lam=1.0)
    print(f"  {'feature':>8}  {'true':>6}  {'Lasso':>8}  {'ridge':>8}")
    for j in range(p):
        print(f"  {j:>8}  {true_coef[j]:>6.1f}  {coef[j]:>8.3f}  {rcoef[j]:>8.3f}")
    print(f"\n  Nonzero coefficients: Lasso {n_nonzero(coef)}, ridge {n_nonzero(rcoef)} (all).")
    print("  Lasso zeros every irrelevant feature and keeps only the true three; ridge shrinks")
    print("  everything but leaves it all nonzero -- no selection.\n")

    print("  Coefficient path (nonzero count shrinks as the L1 penalty grows):")
    print(f"    {'lambda':>8}  {'nonzero':>8}")
    for lam, _, _, nz in lasso_path(X, y, [0.005, 0.02, 0.05, 0.15, 0.4, 1.0]):
        print(f"    {lam:>8.3f}  {nz:>8}")

    _svg(os.path.join(outdir, "lasso.svg"), X, y, true_active)
    print(f"\n  wrote {os.path.join(outdir, 'lasso.svg')}")


def _svg(path, X, y, true_active, width=760, height=400):
    p = len(X[0])
    lambdas = [0.001 * (1.35 ** k) for k in range(28)]
    path_data = lasso_path(X, y, lambdas)
    # coef[j] over lambda
    import math
    coefs_over = [[row[2][j] for row in path_data] for j in range(p)]
    cmax = max(abs(c) for col in coefs_over for c in col) or 1

    ox, oy, ow, oh = 55, 55, width - 100, height - 100
    lx = [math.log10(l) for l in lambdas]

    def px(i):
        return ox + ow * (lx[i] - lx[0]) / (lx[-1] - lx[0])

    def py(v):
        return oy + oh / 2 * (1 - v / cmax)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Lasso coefficient paths: true features stay nonzero, noise features shrink to zero</text>',
    ]
    parts.append(f'<line x1="{ox}" y1="{py(0):.1f}" x2="{ox+ow}" y2="{py(0):.1f}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    for j in range(p):
        col = "#06d6a0" if j in true_active else "#484f58"
        w = 2 if j in true_active else 1
        pts = " ".join(f"{px(i):.1f},{py(coefs_over[j][i]):.1f}" for i in range(len(lambdas)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}"/>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+20:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">log10 lambda (increasing penalty ->)</text>')
    parts.append(f'<text x="{ox+ow-160}" y="{oy+14}" fill="#06d6a0" font-size="10">true features</text>')
    parts.append(f'<text x="{ox+ow-160}" y="{oy+29}" fill="#484f58" font-size="10">noise features</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: barycentric Lagrange interpolation and Runge's phenomenon.

Interpolates Runge's function 1/(1+25x^2) at equally spaced and Chebyshev nodes, shows the equispaced
error exploding near the ends while Chebyshev converges, and draws both interpolants against the true
function.

    python examples/barycentric_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from barycentric import (  # noqa: E402
    chebyshev_interpolant,
    equispaced_interpolant,
    max_error,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Barycentric Lagrange interpolation: stable O(n) evaluation, and Runge's trap\n")

    def runge(x):
        return 1.0 / (1 + 25 * x * x)

    print("  Interpolating Runge's function 1/(1+25x^2) on [-1, 1]:\n")
    print(f"  {'degree':>7}  {'equispaced max error':>21}  {'Chebyshev max error':>20}")
    for deg in [4, 8, 12, 16, 20, 24]:
        eq = max_error(equispaced_interpolant(runge, deg, -1, 1), runge, -1, 1)
        cheb = max_error(chebyshev_interpolant(runge, deg, -1, 1), runge, -1, 1)
        print(f"  {deg:>7}  {eq:>21.3e}  {cheb:>20.3e}")

    print("\n  Equally spaced nodes -> Runge's phenomenon: the interpolant oscillates wildly near the")
    print("  ends and the error GROWS with degree. Chebyshev nodes (clustered at the ends) converge")
    print("  geometrically -- same polynomial degree, opposite fate, decided only by node placement.")

    # smooth function converges fast either way, show it too
    def smooth(x):
        return math.exp(-x) * math.cos(4 * x)
    print("\n  A smooth analytic function exp(-x)cos(4x):")
    for deg in [8, 16, 24]:
        cheb = max_error(chebyshev_interpolant(smooth, deg, -1, 1), smooth, -1, 1)
        print(f"    Chebyshev degree {deg:>2}: max error {cheb:.2e}")

    _svg(os.path.join(outdir, "barycentric.svg"), runge)
    print(f"\n  wrote {os.path.join(outdir, 'barycentric.svg')}")


def _svg(path, runge, width=760, height=420):
    a, b = -1.0, 1.0
    deg = 12
    eq = equispaced_interpolant(runge, deg, a, b)
    cheb = chebyshev_interpolant(runge, deg, a, b)
    xs = [a + (b - a) * k / 400 for k in range(401)]

    true_y = [runge(x) for x in xs]
    eq_y = [eq(x) for x in xs]
    cheb_y = [cheb(x) for x in xs]

    ox, oy, ow, oh = 50, 50, width - 90, height - 100
    # clamp y range so the equispaced overshoot doesn't crush the plot
    ymin, ymax = -0.5, 2.0

    def px(x):
        return ox + ow * (x - a) / (b - a)

    def py(y):
        yy = min(ymax, max(ymin, y))
        return oy + oh * (1 - (yy - ymin) / (ymax - ymin))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Degree-{deg} interpolation of Runge 1/(1+25x^2): equispaced oscillates, Chebyshev fits</text>',
    ]
    # zero line
    parts.append(f'<line x1="{ox}" y1="{py(0):.1f}" x2="{ox+ow}" y2="{py(0):.1f}" '
                 f'stroke="#30363d" stroke-width="1"/>')
    # true function
    tp = " ".join(f"{px(xs[i]):.1f},{py(true_y[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{tp}" fill="none" stroke="#8b949e" stroke-width="2.5" '
                 f'stroke-dasharray="5 3"/>')
    # equispaced (red, oscillates)
    ep = " ".join(f"{px(xs[i]):.1f},{py(eq_y[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{ep}" fill="none" stroke="#ff6b6b" stroke-width="1.6"/>')
    # chebyshev (green, hugs the truth)
    cp = " ".join(f"{px(xs[i]):.1f},{py(cheb_y[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<polyline points="{cp}" fill="none" stroke="#06d6a0" stroke-width="1.6"/>')
    # legend
    parts.append(f'<text x="{ox+ow-160}" y="{oy+14}" fill="#8b949e" font-size="10">true (dashed)</text>')
    parts.append(f'<text x="{ox+ow-160}" y="{oy+29}" fill="#ff6b6b" font-size="10">equispaced</text>')
    parts.append(f'<text x="{ox+ow-160}" y="{oy+44}" fill="#06d6a0" font-size="10">Chebyshev</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

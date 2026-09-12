"""Demo: Chebyshev approximation and the Runge phenomenon it cures.

Shows Chebyshev interpolation converging geometrically on smooth functions, then contrasts Chebyshev
vs equispaced interpolation on Runge's function -- where equispaced blows up and Chebyshev stays
bounded. Draws both fits over Runge's function.

    python examples/chebyshev_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chebyshev import ChebyshevInterpolant, equispaced_interpolate, cheb_nodes  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Chebyshev approximation: near-optimal polynomial fits\n")

    # geometric convergence on a smooth function
    print("  Approximating exp(x) on [-1,1] -- error falls geometrically with degree:")
    for deg in [2, 4, 6, 8, 12, 16]:
        ci = ChebyshevInterpolant(math.exp, deg, -1, 1)
        print(f"    degree {deg:2d}: max error {ci.max_error(math.exp):.2e}")

    # the Runge phenomenon
    runge = lambda x: 1.0 / (1 + 25 * x * x)
    print("\n  Runge's function 1/(1+25x^2) -- Chebyshev vs equally-spaced nodes:")
    print(f"    {'degree':>7}  {'Chebyshev':>12}  {'equispaced':>12}")
    for deg in [8, 12, 16, 20, 24]:
        cheb = ChebyshevInterpolant(runge, deg, -1, 1)
        equi = equispaced_interpolate(runge, deg, -1, 1)
        cheb_err = cheb.max_error(runge)
        equi_err = max(abs(equi(-1 + 2 * i / 500) - runge(-1 + 2 * i / 500)) for i in range(501))
        print(f"    {deg:>7}  {cheb_err:>12.4f}  {equi_err:>12.2f}")
    print("    equispaced interpolation EXPLODES near the ends; Chebyshev nodes (clustered there)")
    print("    tame it -- error spreads evenly across the interval (equioscillation).")

    print("\n  Chebyshev nodes are the projections of equally spaced points on a circle onto the")
    print("  interval, clustered at the ends. Expanding in Chebyshev polynomials T_n and evaluating")
    print("  by Clenshaw recurrence gives a fit within a small factor of the best possible polynomial.")

    _svg(os.path.join(outdir, "chebyshev.svg"), runge)
    print(f"\n  wrote {os.path.join(outdir, 'chebyshev.svg')}")


def _svg(path, runge, width=760, height=430):
    deg = 16
    cheb = ChebyshevInterpolant(runge, deg, -1, 1)
    equi = equispaced_interpolate(runge, deg, -1, 1)

    m_left, m_bot, m_top, m_right = 50, 55, 80, 40
    pw = width - m_left - m_right
    ph = height - m_top - m_bot

    def px(x):
        return m_left + (x + 1) / 2 * pw

    ymin, ymax = -0.5, 2.0

    def py(y):
        return m_top + ph - (max(min(y, ymax), ymin) - ymin) / (ymax - ymin) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="28" fill="#e6edf3" font-size="18">'
        f'Runge phenomenon: Chebyshev (green) vs equispaced (red), degree {deg}</text>',
        f'<text x="20" y="47" fill="#8b949e" font-size="12">'
        f'gray = the true function; equispaced oscillates wildly at the ends, Chebyshev hugs the curve</text>',
    ]
    # zero line
    parts.append(f'<line x1="{m_left}" y1="{py(0):.1f}" x2="{m_left+pw}" y2="{py(0):.1f}" '
                 f'stroke="#30363d" stroke-width="1"/>')

    def curve(fn, color, wdth):
        pts = " ".join(f"{px(-1 + 2*i/400):.1f},{py(fn(-1 + 2*i/400)):.1f}" for i in range(401))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{wdth}"/>'

    parts.append(curve(runge, "#8b949e", 2.5))
    parts.append(curve(equi, "#ff6b6b", 1.5))
    parts.append(curve(cheb, "#06d6a0", 1.8))

    # Chebyshev nodes marked on the axis
    for x in cheb_nodes(deg + 1, -1, 1):
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(0):.1f}" r="2.5" fill="#06d6a0"/>')

    # legend
    ly = m_top + 6
    for col, name in (("#8b949e", "true"), ("#06d6a0", "Chebyshev"), ("#ff6b6b", "equispaced")):
        parts.append(f'<line x1="{m_left+pw-120}" y1="{ly}" x2="{m_left+pw-100}" y2="{ly}" '
                     f'stroke="{col}" stroke-width="2.5"/>')
        parts.append(f'<text x="{m_left+pw-94}" y="{ly+4}" fill="#e6edf3" font-size="10">{name}</text>')
        ly += 16

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: stellar structure via the Lane-Emden equation.

Solves the polytrope equation for several indices n, prints each star's surface
radius xi_1 and mass factor, and renders the density profiles theta(xi) to SVG.
Higher n means a more centrally condensed star.

    python examples/lane_emden_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lane_emden import solve  # noqa: E402

_COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#4cc9f0", "#8338ec"]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Lane-Emden stellar structure: polytrope index n\n")
    print(f"  {'n':>4}{'surface xi_1':>16}{'mass -xi1^2 theta''':>22}{'meaning':>26}")
    print("  " + "-" * 68)
    meanings = {0.0: "uniform-density sphere", 1.0: "analytic sin(xi)/xi",
                1.5: "convective / white dwarf", 3.0: "Eddington standard model",
                5.0: "infinite radius"}
    curves = []
    for n in (0.0, 1.0, 1.5, 3.0, 5.0):
        xis, th, xi1, mass = solve(n, xi_max=(15.0 if n >= 5 else 20.0))
        curves.append((n, xis, th, xi1))
        xi1s = "inf" if xi1 == float("inf") else f"{xi1:.3f}"
        masss = "n/a" if mass != mass else f"{mass:.3f}"
        print(f"  {n:>4}{xi1s:>16}{masss:>22}{meanings[n]:>26}")

    print("\n  As n rises the star grows more centrally concentrated. n=1 is the")
    print("  exact sin(xi)/xi; n=3 is the Eddington standard model of a star;")
    print("  n=5 has finite mass but infinite radius.")

    _svg(curves, os.path.join(outdir, "lane_emden.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'lane_emden.svg')}")


def _svg(curves, path, size=720, pad=56, xi_max=8.0):
    def sx(xi):
        return pad + min(xi, xi_max) / xi_max * (size - 2 * pad)

    def sy(theta):
        return size - pad - theta * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    y = pad + 8
    for i, (n, xis, th, xi1) in enumerate(curves):
        col = _COLORS[i % len(_COLORS)]
        poly = " ".join(f"{sx(xis[j]):.1f},{sy(th[j]):.1f}"
                        for j in range(len(xis)) if xis[j] <= xi_max)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        parts.append(f'<text x="{size-pad-8}" y="{y}" fill="{col}" font-size="12" '
                     f'text-anchor="end">n = {n}</text>')
        y += 16
    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'Lane-Emden density profiles theta(xi)</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+22}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">scaled radius xi -&gt;</text>')
    parts.append(f'<text x="{pad-8}" y="{pad-6}" fill="#8b949e" font-size="11">'
                 f'theta (scaled density)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()

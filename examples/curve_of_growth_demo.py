"""Demo: the curve of growth.

Prints the equivalent width and regime across many decades of central optical depth,
then draws the classic log-log curve of growth with its three segments -- the linear
rise, the flat saturated plateau, and the square-root damping tail -- labelled.

    python examples/curve_of_growth_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from curve_of_growth import equivalent_width, regime  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Curve of growth: equivalent width vs central optical depth\n")
    print(f"  {'tau0':>12}{'W / dnu_D':>14}{'regime':>14}")
    print("  " + "-" * 40)
    for tau in (0.01, 0.1, 1, 10, 100, 1e3, 1e5, 1e7, 1e9):
        W = equivalent_width(tau)
        print(f"  {tau:>12g}{W:>14.4g}{regime(tau):>14}")

    print("\n  Weak lines grow linearly (W ~ N): every atom adds absorption. Once the")
    print("  core saturates the line is already black, so W barely moves over decades")
    print("  of column density -- the flat part where abundances are hardest to pin")
    print("  down. At enormous columns the Lorentzian damping wings go optically thick")
    print("  and W ~ sqrt(N) again. Matching a measured W to this curve is how stellar")
    print("  abundances are read off absorption spectra.")

    _svg(os.path.join(outdir, "curve_of_growth.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'curve_of_growth.svg')}")


def _svg(path, size=720, pad=70):
    taus = [10 ** (-2 + 0.12 * i) for i in range(0, 101)]   # 1e-2 .. 1e10
    Ws = [equivalent_width(t) for t in taus]
    lx = [math.log10(t) for t in taus]
    ly = [math.log10(W) for W in Ws]
    xmin, xmax = lx[0], lx[-1]
    ymin, ymax = min(ly), max(ly)

    def sx(x):
        return pad + (x - xmin) / (xmax - xmin) * (size - 2 * pad)

    def sy(y):
        return size - pad - (y - ymin) / (ymax - ymin) * (size - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{size-pad}" x2="{size-pad}" y2="{size-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{size-pad}" stroke="#30363d"/>',
    ]
    # colour each segment by regime
    seg = {"linear": "#4dabf7", "saturated": "#ffd43b", "damped": "#ff6b6b"}
    for i in range(len(taus) - 1):
        col = seg[regime(taus[i])]
        parts.append(f'<line x1="{sx(lx[i]):.1f}" y1="{sy(ly[i]):.1f}" '
                     f'x2="{sx(lx[i+1]):.1f}" y2="{sy(ly[i+1]):.1f}" '
                     f'stroke="{col}" stroke-width="2.8"/>')

    parts.append(f'<text x="{sx(-1):.1f}" y="{sy(-0.5):.1f}" fill="#4dabf7" '
                 f'font-size="12">linear W~N</text>')
    parts.append(f'<text x="{sx(3):.1f}" y="{sy(0.9):.1f}" fill="#ffd43b" '
                 f'font-size="12">saturated (flat)</text>')
    parts.append(f'<text x="{sx(7.5):.1f}" y="{sy(2.2):.1f}" fill="#ff6b6b" '
                 f'font-size="12" text-anchor="end">damped W~sqrt(N)</text>')

    parts.append(f'<text x="{pad}" y="34" fill="#e6edf3" font-size="18">'
                 f'The curve of growth</text>')
    parts.append(f'<text x="{pad}" y="52" fill="#8b949e" font-size="12">'
                 f'linear rise -> saturated plateau -> square-root damping tail</text>')
    parts.append(f'<text x="{size-pad}" y="{size-pad+24}" fill="#8b949e" '
                 f'font-size="11" text-anchor="end">log10 central optical depth tau0 -&gt;</text>')
    parts.append(f'<text x="{pad-16}" y="{pad-8}" fill="#8b949e" font-size="11">'
                 f'log10 equivalent width</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()

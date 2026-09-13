"""Demo: Remez minimax approximation of e^x -- the error that rides evenly, +E to -E to +E.

Fits a low-degree polynomial to e^x on [0, 2] three ways -- minimax (Remez), Chebyshev-node
interpolation, and least squares -- and shows the minimax error curve equioscillating n+2 times at a
lower peak than either rival. Draws the three error curves.

    python examples/remez_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from remez import (  # noqa: E402
    remez,
    poly_eval,
    max_error,
    equioscillation_points,
    chebyshev_interp,
    least_squares_poly,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    f = math.exp
    a, b = 0.0, 2.0
    degree = 4

    print("Remez minimax: the best polynomial approximation of e^x on [0, 2]\n")

    mcoef, mlevel, refs = remez(f, a, b, degree)
    ccoef = chebyshev_interp(f, a, b, degree)
    lcoef = least_squares_poly(f, a, b, degree)

    merr = max_error(f, mcoef, a, b)
    cerr = max_error(f, ccoef, a, b)
    lerr = max_error(f, lcoef, a, b)

    print(f"  degree-{degree} fits, worst-case error max|e^x - p(x)|:")
    print(f"    {'method':<26}{'sup error':>12}")
    print(f"    {'minimax (Remez)':<26}{merr:>12.3e}")
    print(f"    {'Chebyshev interpolation':<26}{cerr:>12.3e}   ({cerr / merr:.2f}x worse)")
    print(f"    {'least squares':<26}{lerr:>12.3e}   ({lerr / merr:.2f}x worse)")

    pts = equioscillation_points(f, mcoef, a, b)
    big = [(x, e) for x, e in pts if abs(e) > 0.9 * mlevel]
    print(f"\n  minimax error equioscillates at {len(big)} points (need {degree + 2}):")
    for x, e in big:
        print(f"    x = {x:5.3f}   error = {e:+.6e}")
    print(f"\n  Every peak reaches +/- {mlevel:.3e}, alternating sign -- Chebyshev's")
    print("  equioscillation theorem: no degree-4 polynomial does better.")

    _svg(os.path.join(outdir, "remez.svg"), f, mcoef, ccoef, lcoef, a, b, big, mlevel)
    print(f"\n  wrote {os.path.join(outdir, 'remez.svg')}")


def _svg(path, f, mcoef, ccoef, lcoef, a, b, big, mlevel, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Approximation error e^x - p(x): minimax equioscillates at the lowest peak</text>',
    ]
    ox, oy, ow, oh = 55, 50, width - 90, height - 100

    grid = 400
    xs = [a + (b - a) * k / (grid - 1) for k in range(grid)]

    def err(coef):
        return [f(x) - poly_eval(coef, x) for x in xs]

    em = err(mcoef)
    ec = err(ccoef)
    el = err(lcoef)
    emax = max(max(abs(v) for v in em), max(abs(v) for v in ec), max(abs(v) for v in el))

    def px(x):
        return ox + ow * (x - a) / (b - a)

    def py(e):
        return oy + oh * (0.5 - 0.5 * e / emax)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    # zero axis
    parts.append(f'<line x1="{ox}" y1="{py(0):.1f}" x2="{ox+ow}" y2="{py(0):.1f}" '
                 f'stroke="#30363d" stroke-dasharray="3 3"/>')
    # +/- E band for minimax
    parts.append(f'<line x1="{ox}" y1="{py(mlevel):.1f}" x2="{ox+ow}" y2="{py(mlevel):.1f}" '
                 f'stroke="#4dabf7" stroke-width="1" stroke-dasharray="4 3" stroke-opacity="0.6"/>')
    parts.append(f'<line x1="{ox}" y1="{py(-mlevel):.1f}" x2="{ox+ow}" y2="{py(-mlevel):.1f}" '
                 f'stroke="#4dabf7" stroke-width="1" stroke-dasharray="4 3" stroke-opacity="0.6"/>')

    def poly(vals, color, w=2):
        pts_str = " ".join(f"{px(xs[k]):.1f},{py(vals[k]):.1f}" for k in range(grid))
        parts.append(f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="{w}"/>')

    poly(el, "#8b949e", 1.5)   # least squares
    poly(ec, "#ff922b", 1.5)   # chebyshev
    poly(em, "#4dabf7", 2.5)   # minimax
    # equioscillation markers
    for x, e in big:
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(e):.1f}" r="4" fill="#ffd43b"/>')

    # legend
    ly = oy + 14
    for label, color in [("minimax (Remez)", "#4dabf7"), ("Chebyshev interp", "#ff922b"),
                         ("least squares", "#8b949e")]:
        parts.append(f'<rect x="{ox+ow-150}" y="{ly-9}" width="12" height="4" fill="{color}"/>')
        parts.append(f'<text x="{ox+ow-134}" y="{ly-4}" fill="#e6edf3" font-size="10">{label}</text>')
        ly += 16

    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">x on [{a:.0f}, {b:.0f}]</text>')
    parts.append(f'<text x="{ox-6}" y="{py(mlevel):.0f}" fill="#4dabf7" font-size="9" '
                 f'text-anchor="end">+E</text>')
    parts.append(f'<text x="{ox-6}" y="{py(-mlevel):.0f}" fill="#4dabf7" font-size="9" '
                 f'text-anchor="end">-E</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

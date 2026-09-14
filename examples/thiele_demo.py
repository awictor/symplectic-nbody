"""Demo: Thiele rational interpolation beating polynomials on the Runge function and near a pole.

Fits the Runge function 1/(1+25x^2) with both Thiele's rational interpolant and the equispaced
polynomial interpolant, showing the polynomial's wild edge oscillations against Thiele's exactness,
and reconstructs a function with a genuine pole that no polynomial can. Draws both interpolants
against the truth.

    python examples/thiele_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thiele import ThieleInterpolant, thiele_fit, lagrange_eval, runge  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Thiele's continued-fraction interpolation: rational functions, poles, no Runge blow-up\n")

    # Runge function on 11 equispaced nodes
    xs = [i * 0.2 - 1 for i in range(11)]
    ys = [runge(x) for x in xs]
    t = ThieleInterpolant(xs, ys)

    test = [i * 0.02 - 1 for i in range(101)]
    thiele_err = max(abs(t(x) - runge(x)) for x in test)
    lagrange_err = max(abs(lagrange_eval(xs, ys, x) - runge(x)) for x in test)

    print(f"  Runge function 1/(1+25x^2), 11 equispaced nodes on [-1, 1]:")
    print(f"    max error, Thiele (rational):   {thiele_err:.2e}")
    print(f"    max error, Lagrange (polynomial): {lagrange_err:.3f}")
    print(f"    Thiele is {lagrange_err / max(thiele_err, 1e-16):.0f}x more accurate -- the Runge")
    print(f"    function IS rational, so Thiele reproduces it while the polynomial oscillates.\n")

    # a function with a real pole
    pf = lambda x: 1.0 / (x - 0.35)
    xsp = [-2, -1, 0, 1, 2, 3]
    tp = thiele_fit(pf, xsp)
    print(f"  a function with a pole at x = 0.35: 1/(x - 0.35)")
    print(f"    {'x':>6}{'Thiele':>14}{'true':>14}")
    for x in [0.7, 1.2, -0.5, 2.5]:
        print(f"    {x:>6}{tp(x):>14.6f}{pf(x):>14.6f}")
    print(f"  a polynomial can never reproduce that blow-up; the rational interpolant places the")
    print(f"  pole exactly where it belongs.")

    _svg(os.path.join(outdir, "thiele.svg"), xs, ys, t)
    print(f"\n  wrote {os.path.join(outdir, 'thiele.svg')}")


def _svg(path, xs, ys, t, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Runge function (green), Thiele rational fit (blue), polynomial fit (orange, oscillates)</text>',
    ]
    ox, oy, ow, oh = 45, 50, width - 80, height - 100
    xmin, xmax = -1.0, 1.0
    test = [xmin + (xmax - xmin) * k / 300 for k in range(301)]
    truth = [runge(x) for x in test]
    tvals = [t(x) for x in test]
    pvals = [lagrange_eval(xs, ys, x) for x in test]
    # clamp polynomial for display
    ymin, ymax = -0.5, 1.6

    def sx(x):
        return ox + ow * (x - xmin) / (xmax - xmin)

    def sy(y):
        yy = max(min(y, ymax), ymin)
        return oy + oh * (1 - (yy - ymin) / (ymax - ymin))

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')

    def poly(vals, color, w=2):
        pts = " ".join(f"{sx(test[k]):.1f},{sy(vals[k]):.1f}" for k in range(len(test)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}"/>')

    poly(pvals, "#ff922b", 1.5)   # polynomial (oscillating)
    poly(truth, "#06d6a0", 2.5)   # truth
    poly(tvals, "#4dabf7", 1.5)   # thiele (on top of truth)
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="#e6edf3"/>')
    parts.append(f'<rect x="{ox+ow-160}" y="{oy+6}" width="12" height="3" fill="#06d6a0"/>')
    parts.append(f'<text x="{ox+ow-144}" y="{oy+10}" fill="#e6edf3" font-size="10">Runge (truth)</text>')
    parts.append(f'<rect x="{ox+ow-160}" y="{oy+22}" width="12" height="3" fill="#4dabf7"/>')
    parts.append(f'<text x="{ox+ow-144}" y="{oy+26}" fill="#e6edf3" font-size="10">Thiele rational</text>')
    parts.append(f'<rect x="{ox+ow-160}" y="{oy+38}" width="12" height="3" fill="#ff922b"/>')
    parts.append(f'<text x="{ox+ow-144}" y="{oy+42}" fill="#e6edf3" font-size="10">polynomial</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

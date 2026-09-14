"""Demo: Richardson extrapolation turning a second-order derivative formula into machine precision.

Shows the Richardson tableau for the derivative of sin at x=1: the first column (raw central
differences) is only second order, but each extrapolation column cancels the next error term, and the
diagonal races to machine precision. Also extrapolates the limit (1+h)^(1/h) -> e. Draws the tableau
error decay.

    python examples/richardson_extrapolation_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from richardson_extrapolation import (  # noqa: E402
    derivative_tableau, derivative, limit, _central_difference,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Richardson extrapolation: a 2nd-order formula driven to machine precision\n")

    x = 1.0
    true = math.cos(x)
    T = derivative_tableau(math.sin, x, h0=0.5, levels=6)

    print(f"  d/dx sin(x) at x=1, true value cos(1) = {true:.15f}\n")
    print(f"  Richardson tableau (each column cancels the next error order):")
    print(f"    {'h':>8}  raw central diff, then extrapolations ->")
    h = 0.5
    for i, row in enumerate(T):
        errs = "  ".join(f"{abs(v - true):.1e}" for v in row)
        print(f"    {h:>8.4f}  {errs}")
        h /= 2
    print(f"\n  first column (raw): error ~ {abs(T[0][0]-true):.1e}, only O(h^2)")
    print(f"  bottom-right (extrapolated): error ~ {abs(T[-1][-1]-true):.1e}, near machine epsilon")
    print(f"  -- from the SAME central-difference evaluations, no smaller h needed.\n")

    # limit example
    e_est = limit(lambda hh: (1 + hh) ** (1 / hh), h0=1.0, p=1, t=2, levels=12)
    print(f"  limit (1+h)^(1/h) as h->0:")
    print(f"    raw at h=1:   {(1+1)**(1/1):.6f}")
    print(f"    extrapolated: {e_est:.12f}   (true e = {math.e:.12f})")

    print(f"\n  This error cancellation is the engine inside Romberg integration and")
    print(f"  Bulirsch-Stoer ODE solving -- cheap low-order evaluations, high-order result.")

    _svg(os.path.join(outdir, "richardson_extrapolation.svg"), T, true)
    print(f"\n  wrote {os.path.join(outdir, 'richardson_extrapolation.svg')}")


def _svg(path, T, true, width=760, height=400):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Richardson tableau error by column: each extrapolation kills the next error order</text>',
    ]
    ox, oy, ow, oh = 60, 55, width - 110, height - 110
    n = len(T)
    # error of each column's diagonal-ish entries
    all_errs = [max(abs(v - true), 1e-17) for row in T for v in row]
    lo = math.log10(min(all_errs))
    hi = math.log10(max(all_errs))

    def py(e):
        lg = math.log10(max(abs(e), 1e-17))
        return oy + oh * (1 - (lg - lo) / (hi - lo + 1e-12))

    def px(i):
        return ox + ow * i / (n - 1)

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    d = math.floor(lo)
    while d <= hi:
        y = py(10 ** d)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+ow}" y2="{y:.1f}" stroke="#161b22"/>')
        parts.append(f'<text x="{ox-6}" y="{y+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">1e{int(d)}</text>')
        d += 3
    colors = ["#8b949e", "#4dabf7", "#06d6a0", "#ffd43b", "#ff922b", "#ff6b6b"]
    # each column k: plot error of T[i][k] for i>=k
    for k in range(n):
        pts = []
        for i in range(k, n):
            pts.append(f"{px(i):.1f},{py(T[i][k] - true):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[k % len(colors)]}" '
                     f'stroke-width="1.8"/>')
        for i in range(k, n):
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(T[i][k]-true):.1f}" r="2.5" '
                         f'fill="{colors[k % len(colors)]}"/>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+22:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">row (smaller h ->); gray = raw central diff, each colour a '
                 f'further extrapolation column</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

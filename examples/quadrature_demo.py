"""Demo: numerical quadrature -- integrating what algebra cannot.

Integrates a test function five ways, reports the error at a fixed sample budget, and shows the
convergence orders (trapezoid O(h^2), Simpson O(h^4), Romberg/Gauss much faster). Draws the
error-versus-samples curves on a log-log plot where each method's slope is its order.

    python examples/quadrature_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quadrature import (trapezoid, simpson, romberg,  # noqa: E402
                        adaptive_simpson, gauss_legendre)


# a smooth test integral with a known exact value
def f(x):
    return math.exp(x) * math.cos(x)


A, B = 0.0, math.pi
EXACT = (math.exp(math.pi) * (math.sin(math.pi) + math.cos(math.pi))
         - (math.sin(0) + math.cos(0))) / 2      # integral of e^x cos x


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Numerical quadrature: integral of e^x cos x on [0, pi]\n")
    print(f"  exact value = {EXACT:.12f}\n")
    print(f"  {'method':>18}{'estimate':>20}{'error':>12}")
    for name, val in (("trapezoid (n=100)", trapezoid(f, A, B, 100)),
                      ("Simpson (n=100)", simpson(f, A, B, 100)),
                      ("Gauss-Legendre (8p)", gauss_legendre(f, A, B, 8)),
                      ("adaptive Simpson", adaptive_simpson(f, A, B)),
                      ("Romberg", romberg(f, A, B))):
        print(f"  {name:>18}{val:>20.12f}{abs(val - EXACT):>12.1e}")

    print("\n  Convergence order (error as the sample count n doubles):")
    print(f"  {'n':>6}{'trapezoid':>14}{'ratio':>8}{'Simpson':>14}{'ratio':>8}")
    prev_t = prev_s = None
    for n in (16, 32, 64, 128, 256):
        et = abs(trapezoid(f, A, B, n) - EXACT)
        es = abs(simpson(f, A, B, n) - EXACT)
        rt = f"{prev_t / et:.1f}x" if prev_t else "-"
        rs = f"{prev_s / es:.1f}x" if prev_s else "-"
        print(f"  {n:>6}{et:>14.2e}{rt:>8}{es:>14.2e}{rs:>8}")
        prev_t, prev_s = et, es
    print("\n  Trapezoid halves-the-step-quarters-the-error (O(h^2)); Simpson cuts it 16-fold")
    print("  (O(h^4)); Romberg and Gauss-Legendre reach machine precision in a few evaluations.")
    print("  It powers everything from physics simulations to option pricing to Bayesian evidence.")

    _svg(os.path.join(outdir, "quadrature.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'quadrature.svg')}")


def _svg(path, w=760, h=390):
    # error vs number of function evaluations, log-log
    ns = [2 ** k for k in range(2, 12)]
    trap_e = [max(abs(trapezoid(f, A, B, n) - EXACT), 1e-17) for n in ns]
    simp_e = [max(abs(simpson(f, A, B, n) - EXACT), 1e-17) for n in ns]
    gauss_e = [max(abs(gauss_legendre(f, A, B, max(1, n // 5)) - EXACT), 1e-17) for n in ns]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Quadrature convergence: steeper slope = higher order</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'error vs samples (log-log): trapezoid slope -2, Simpson -4, Gauss far steeper</text>',
    ]

    x0, x1 = 60, w - 40
    y0, y1 = h - 55, 62
    lo_n, hi_n = math.log2(ns[0]), math.log2(ns[-1])
    emin, emax = -16.0, 1.0

    def X(n):
        return x0 + (math.log2(n) - lo_n) / (hi_n - lo_n) * (x1 - x0)

    def Y(e):
        le = max(emin, min(emax, math.log10(e)))
        return y0 - (le - emin) / (emax - emin) * (y0 - y1)

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    for e in range(-16, 2, 4):
        yy = Y(10.0 ** e)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-6:.1f}" y="{yy+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">1e{e}</text>')
    ly = y1
    for series, col, lab in ((trap_e, "#ff6b6b", "trapezoid O(h^2)"),
                             (simp_e, "#ffd43b", "Simpson O(h^4)"),
                             (gauss_e, "#06d6a0", "Gauss-Legendre")):
        pts = " ".join(f"{X(n):.1f},{Y(e):.1f}" for n, e in zip(ns, series))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        for n, e in zip(ns, series):
            parts.append(f'<circle cx="{X(n):.1f}" cy="{Y(e):.1f}" r="2" fill="{col}"/>')
        parts.append(f'<rect x="{x0+8}" y="{ly}" width="9" height="9" fill="{col}"/>'
                     f'<text x="{x0+21}" y="{ly+8}" fill="#e6edf3" font-size="9">{lab}</text>')
        ly += 14
    for n in (4, 64, 2048):
        parts.append(f'<text x="{X(n):.1f}" y="{y0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">samples n (log) -> absolute error (log)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

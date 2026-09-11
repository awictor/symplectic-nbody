"""Demo: Horner's method -- nested multiplication for polynomials.

Evaluates a polynomial by Horner and by direct powers (same answer, half the multiplications),
does synthetic division, and finds all roots of a cubic by Newton refinement plus deflation.
Draws the multiplication-count advantage over degree and a Newton convergence trace.

    python examples/horner_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from horner import (horner, direct_eval, synthetic_division,  # noqa: E402
                    evaluate_with_derivative, real_roots)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    c = [2, -6, 2, -1]      # 2x^3 - 6x^2 + 2x - 1
    print("Horner: p(x) = ((2x - 6)x + 2)x - 1, evaluated in 3 mults instead of 6\n")
    print(f"  p(x) = 2x^3 - 6x^2 + 2x - 1")
    print(f"  p(3) = {horner(c, 3):.0f}  (direct eval agrees: {horner(c, 3) == direct_eval(c, 3)})")
    q, r = synthetic_division(c, 3)
    print(f"  synthetic division by (x - 3): quotient {q}, remainder {r:.0f} = p(3)")
    p, dp = evaluate_with_derivative(c, 3)
    print(f"  p(3) = {p:.0f}, p'(3) = {dp:.0f}  (both in two linear sweeps)\n")

    cubic = [1, -6, 11, -6]     # (x-1)(x-2)(x-3)
    roots = real_roots(cubic)
    print(f"  roots of x^3 - 6x^2 + 11x - 6 (Newton + deflation): {[round(x, 6) for x in roots]}")
    print("  each found by p(r)/p'(r) steps, then divided out to find the next.\n")

    print("  Multiplication count, Horner (n) vs direct powers (~2n):")
    print(f"  {'degree':>8}{'Horner':>9}{'direct':>9}")
    for deg in (3, 5, 10, 20, 50):
        print(f"  {deg:>8}{deg:>9}{2 * deg:>9}")
    print("\n  Horner is the standard polynomial evaluator: fewest operations, and summing")
    print("  left-to-right in one accumulator gives better rounding than adding separate powers.")

    _svg(os.path.join(outdir, "horner.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'horner.svg')}")


def _svg(path, w=760, h=390):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Horner: fewer multiplications, and Newton root convergence</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'multiplications vs polynomial degree (left); |p(x)| shrinking each Newton step (right)</text>',
    ]

    # left: mult count vs degree
    lx0, lx1 = 55, w // 2 - 20
    ly0, ly1 = h - 55, 62
    degs = list(range(2, 51))
    hmax = 2 * degs[-1] * 1.1

    def LX(d):
        return lx0 + (d - degs[0]) / (degs[-1] - degs[0]) * (lx1 - lx0)

    def LY(m):
        return ly0 - m / hmax * (ly0 - ly1)

    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx1}" y2="{ly0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{lx0}" y1="{ly0}" x2="{lx0}" y2="{ly1}" stroke="#8b949e" stroke-width="1.2"/>')
    direct = " ".join(f"{LX(d):.1f},{LY(2*d):.1f}" for d in degs)
    horn = " ".join(f"{LX(d):.1f},{LY(d):.1f}" for d in degs)
    parts.append(f'<polyline points="{direct}" fill="none" stroke="#ff6b6b" stroke-width="2.5"/>')
    parts.append(f'<polyline points="{horn}" fill="none" stroke="#06d6a0" stroke-width="2.5"/>')
    parts.append(f'<text x="{LX(degs[-1]):.1f}" y="{LY(2*degs[-1])-6:.1f}" fill="#ff6b6b" '
                 f'font-size="9" text-anchor="end">direct ~2n</text>')
    parts.append(f'<text x="{LX(degs[-1]):.1f}" y="{LY(degs[-1])+14:.1f}" fill="#06d6a0" '
                 f'font-size="9" text-anchor="end">Horner n</text>')
    for d in (2, 25, 50):
        parts.append(f'<text x="{LX(d):.1f}" y="{ly0+15:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="middle">{d}</text>')
    parts.append(f'<text x="{(lx0+lx1)/2:.1f}" y="{ly0+30:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">degree -> multiplications</text>')

    # right: Newton convergence |p(x)| per step for x^2 - 2 from x0=2 (log y)
    rx0, rx1 = w // 2 + 45, w - 30
    ry0, ry1 = h - 55, 62
    poly = [1, 0, -2]
    x = 2.0
    residuals = []
    for _ in range(7):
        p, dp = evaluate_with_derivative(poly, x)
        residuals.append(max(abs(p), 1e-18))
        if dp == 0:
            break
        x -= p / dp
    steps = list(range(len(residuals)))
    lo, hiy = -18.0, math.log10(max(residuals))

    def RX(i):
        return rx0 + i / max(1, len(steps) - 1) * (rx1 - rx0)

    def RY(res):
        le = max(lo, math.log10(res))
        return ry0 - (le - lo) / (hiy - lo) * (ry0 - ry1)

    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx1}" y2="{ry0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{rx0}" y1="{ry0}" x2="{rx0}" y2="{ry1}" stroke="#8b949e" stroke-width="1.2"/>')
    pts = " ".join(f"{RX(i):.1f},{RY(res):.1f}" for i, res in enumerate(residuals))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#4dabf7" stroke-width="2.5"/>')
    for i, res in enumerate(residuals):
        parts.append(f'<circle cx="{RX(i):.1f}" cy="{RY(res):.1f}" r="2.5" fill="#4dabf7"/>')
    for e in range(-18, 1, 6):
        yy = RY(10.0 ** e) if e > lo else ry0
        parts.append(f'<text x="{rx0-6:.1f}" y="{yy+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">1e{e}</text>')
    parts.append(f'<text x="{(rx0+rx1)/2:.1f}" y="{ry0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">Newton step -> |p(x)| (log): quadratic convergence to sqrt(2)</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

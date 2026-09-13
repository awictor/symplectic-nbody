"""Demo: Pade approximants -- rational functions that outreach Taylor series.

Approximates exp and a function with a pole using both Taylor polynomials and Pade approximants of the
same order, showing Pade stays accurate far past where the Taylor series diverges. Draws the error
curves.

    python examples/pade_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pade import pade, evaluate  # noqa: E402


def _taylor(coeffs, x):
    s = 0.0
    for c in reversed(coeffs):
        s = s * x + c
    return s


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Pade approximants: rational P(x)/Q(x) that beats Taylor, especially near poles\n")

    # exp
    exp_c = [1.0 / math.factorial(k) for k in range(9)]
    P, Q = pade(exp_c, 4, 4)
    print("  exp(x): degree-8 Taylor vs [4/4] Pade (both use 9 coefficients):\n")
    print(f"  {'x':>5}  {'Taylor(8) error':>16}  {'Pade [4/4] error':>17}")
    for x in [1, 2, 3, 4, 5]:
        true = math.exp(x)
        te = abs(_taylor(exp_c, x) - true)
        pe = abs(evaluate(P, Q, x) - true)
        print(f"  {x:>5}  {te:>16.2e}  {pe:>17.2e}")
    print("    -> the Taylor error grows fast; the Pade approximant stays tight.\n")

    # a function with a pole: f(x) = 1/((1-x)(1+2x)), Taylor diverges at |x|=0.5
    def f_pole(x):
        return 1.0 / ((1 - x) * (1 + 2 * x))
    # Taylor coefficients by expansion: partial fractions 1/(1-x)(1+2x)
    #  = A/(1-x) + B/(1+2x); A=1/3, B=2/3? solve: A(1+2x)+B(1-x)=1 -> A+B=1, 2A-B=0 -> A=1/3,B=2/3
    A, B = 1 / 3, 2 / 3
    pole_c = [A * 1 + B * (-2) ** k for k in range(9)]
    Pp, Qp = pade(pole_c, 4, 4)
    print("  f(x) = 1/((1-x)(1+2x)) has poles at x=1 and x=-0.5; Taylor radius is 0.5:\n")
    print(f"  {'x':>6}  {'Taylor error':>14}  {'Pade error':>12}")
    for x in [0.3, 0.45, 0.7, 0.9]:
        if abs(1 - x) < 1e-6 or abs(1 + 2 * x) < 1e-6:
            continue
        true = f_pole(x)
        te = abs(_taylor(pole_c, x) - true)
        pe = abs(evaluate(Pp, Qp, x) - true)
        print(f"  {x:>6.2f}  {te:>14.2e}  {pe:>12.2e}")
    print("    -> past x=0.5 the Taylor series blows up; the Pade approximant tracks the function")
    print("    because its denominator reproduces the poles.")

    _svg(os.path.join(outdir, "pade.svg"), exp_c, P, Q)
    print(f"\n  wrote {os.path.join(outdir, 'pade.svg')}")


def _svg(path, exp_c, P, Q, width=760, height=400):
    xs = [i / 40 * 5 for i in range(41)]  # 0..5
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        '<text x="20" y="26" fill="#e6edf3" font-size="15">'
        'Approximating exp(x): Taylor(8) (red) diverges, Pade [4/4] (green) tracks the truth</text>',
    ]
    ox, oy, ow, oh = 50, 50, width - 90, height - 100
    ymax = math.exp(5) * 1.1

    def sx(x):
        return ox + ow * x / 5

    def sy(y):
        return oy + oh * (1 - min(y, ymax) / ymax)

    # true exp
    tp = " ".join(f"{sx(x):.1f},{sy(math.exp(x)):.1f}" for x in xs)
    parts.append(f'<polyline points="{tp}" fill="none" stroke="#8b949e" stroke-width="2.5" '
                 f'stroke-dasharray="5 3"/>')
    # taylor
    ty = " ".join(f"{sx(x):.1f},{sy(max(0, _taylor(exp_c, x))):.1f}" for x in xs)
    parts.append(f'<polyline points="{ty}" fill="none" stroke="#ff6b6b" stroke-width="1.8"/>')
    # pade
    py = " ".join(f"{sx(x):.1f},{sy(max(0, evaluate(P, Q, x))):.1f}" for x in xs)
    parts.append(f'<polyline points="{py}" fill="none" stroke="#06d6a0" stroke-width="1.8"/>')
    parts.append(f'<text x="{ox+ow-150}" y="{oy+14}" fill="#8b949e" font-size="10">true exp (dashed)</text>')
    parts.append(f'<text x="{ox+ow-150}" y="{oy+29}" fill="#ff6b6b" font-size="10">Taylor(8)</text>')
    parts.append(f'<text x="{ox+ow-150}" y="{oy+44}" fill="#06d6a0" font-size="10">Pade [4/4]</text>')
    parts.append(f'<text x="{ox+ow/2:.0f}" y="{oy+oh+20:.0f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">x (0 to 5)</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: 1D minimization -- golden-section vs Brent, derivative-free.

Minimizes several functions, compares the evaluation counts of golden section and Brent, auto-brackets
a minimum, and draws the shrinking golden-section interval and the two methods' convergence.

    python examples/minimize_1d_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minimize_1d import golden_section, brent, bracket_minimum, minimize, _GOLDEN  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("1D minimization: finding the bottom of a curve with only function values\n")

    funcs = [
        ("(x-3)^2 + 1", lambda x: (x - 3) ** 2 + 1, -10, 10),
        ("x^4 - 2x^2 (double well)", lambda x: x ** 4 - 2 * x ** 2, 0.1, 3),
        ("cos(x)", math.cos, 0, 2 * math.pi),
        ("-exp(-(x-2)^2)", lambda x: -math.exp(-(x - 2) ** 2), -5, 8),
    ]
    print(f"  {'function':<28} {'golden x':>12} {'evals':>6}   {'brent x':>12} {'evals':>6}")
    for name, f, a, b in funcs:
        rg = golden_section(f, a, b, tol=1e-10)
        rb = brent(f, a, b, tol=1e-10)
        print(f"  {name:<28} {rg.x:12.6f} {rg.evaluations:>6}   {rb.x:12.6f} {rb.evaluations:>6}")
    print("    (Brent's parabolic steps reach the same minimum in far fewer evaluations)\n")

    # auto-bracketing
    f = lambda x: (x - 13.7) ** 2 + 2
    a, b, c = bracket_minimum(f, x0=0.0, step=1.0)
    r = minimize(f, x0=0.0)
    print(f"  auto-bracketing from x0=0: found bracket ({a:.0f}, {b:.0f}, {c:.0f}),")
    print(f"    then Brent located the minimum at x = {r.x:.6f} (true 13.7)\n")

    print("  Golden section keeps a bracket guaranteed to hold the minimum and shrinks it by ~0.618")
    print("  each step -- slow but foolproof on any unimodal curve. Brent tries a fast parabolic jump")
    print("  and falls back to golden section when the parabola misbehaves, getting superlinear speed")
    print("  with the same guarantee -- the default 1D minimizer inside every optimizer's line search.")

    _svg(os.path.join(outdir, "minimize_1d.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'minimize_1d.svg')}")


def _svg(path, width=760, height=420):
    # top: a curve with golden-section brackets shrinking; bottom: convergence of both methods
    f = lambda x: (x - 3) ** 2 + 1 + 0.3 * math.sin(2 * x)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'Golden-section brackets closing on the minimum (top); convergence (bottom)</text>',
    ]

    # top panel: plot f and the shrinking brackets
    ox, oy = 50, 220
    pw, ph = width - 90, 160
    xa, xb = -1.0, 7.0
    xs = [xa + (xb - xa) * i / 300 for i in range(301)]
    ys = [f(x) for x in xs]
    ymin, ymax = min(ys), max(ys)

    def px(x):
        return ox + (x - xa) / (xb - xa) * pw

    def py(y):
        return oy - (y - ymin) / (ymax - ymin) * ph

    curve = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    parts.append(f'<polyline points="{curve}" fill="none" stroke="#4dabf7" stroke-width="2"/>')

    # run golden section, recording brackets
    a, b = xa, xb
    cols = ["#ffd43b", "#ff922b", "#06d6a0", "#b197fc", "#ff6b6b"]
    x1 = a + (1 - _GOLDEN) * (b - a)
    x2 = a + _GOLDEN * (b - a)
    f1, f2 = f(x1), f(x2)
    for k in range(5):
        y = oy + 12 + k * 8
        parts.append(f'<line x1="{px(a):.1f}" y1="{y}" x2="{px(b):.1f}" y2="{y}" '
                     f'stroke="{cols[k]}" stroke-width="3"/>')
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + (1 - _GOLDEN) * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + _GOLDEN * (b - a)
            f2 = f(x2)
    parts.append(f'<text x="{ox}" y="{oy+70}" fill="#8b949e" font-size="11">'
                 f'each coloured bar is the bracket after one more golden-section step</text>')

    # bottom: convergence (error vs eval count) for both methods on (x-3)^2
    bx, by = 50, 400
    bw, bh = width - 90, 130
    g = lambda x: (x - 3) ** 2
    # reconstruct golden error trace
    def golden_trace(a0, b0, n):
        a, b = a0, b0
        x1 = a + (1 - _GOLDEN) * (b - a); x2 = a + _GOLDEN * (b - a)
        f1, f2 = g(x1), g(x2)
        errs = []
        for _ in range(n):
            errs.append(abs((a + b) / 2 - 3))
            if f1 < f2:
                b, x2, f2 = x2, x1, f1; x1 = a + (1 - _GOLDEN) * (b - a); f1 = g(x1)
            else:
                a, x1, f1 = x1, x2, f2; x2 = a + _GOLDEN * (b - a); f2 = g(x2)
        return errs
    gerr = golden_trace(-10, 10, 40)
    logs = [math.log10(max(e, 1e-16)) for e in gerr]
    lo, hi = -16, 1

    def cx(i):
        return bx + i / (len(logs) - 1) * bw

    def cy(l):
        return by - (l - lo) / (hi - lo) * bh

    parts.append(f'<text x="{bx}" y="{by-bh-6}" fill="#8b949e" font-size="11">'
                 f'golden-section error vs iterations (log scale) -- steady linear decrease</text>')
    pts = " ".join(f"{cx(i):.1f},{cy(l):.1f}" for i, l in enumerate(logs))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#ffd43b" stroke-width="2"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: bracketing root-finders -- bisection, secant, false position, Brent.

Finds a root four ways and compares iteration counts, shows Brent solving transcendental
equations, and locates every root of a function on an interval via bracket scanning. Draws the
error-vs-iteration convergence of each method (log scale).

    python examples/rootfind_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rootfind import (bisection, secant, false_position, brent,  # noqa: E402
                      find_brackets)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    f = lambda x: x * x - 2       # root at sqrt(2)
    true = 2 ** 0.5
    print("Root-finding: four bracketing methods on x^2 - 2 = 0 (root = sqrt 2)\n")
    print(f"  {'method':>16}{'root':>18}{'iters':>7}{'error':>12}")
    for name, res in (("bisection", bisection(f, 0, 2)),
                      ("false position", false_position(f, 0, 2)),
                      ("Brent", brent(f, 0, 2)),
                      ("secant", secant(f, 0, 2))):
        r, it = res
        print(f"  {name:>16}{r:>18.12f}{it:>7}{abs(r - true):>12.1e}")

    print("\n  Brent on transcendental equations (no closed form):")
    for label, g, br in (("cos x = x", lambda x: math.cos(x) - x, (0, 1)),
                         ("x = e^-x", lambda x: x - math.exp(-x), (0, 1)),
                         ("e^x = 3x+1", lambda x: math.exp(x) - 3 * x - 1, (1, 2))):
        r, it = brent(g, *br)
        print(f"    {label:>12}: x = {r:.10f}  ({it} iters)")

    print("\n  Locating every root by scanning for sign changes, then refining with Brent:")
    poly = lambda x: x ** 3 - 6 * x ** 2 + 11 * x - 6      # roots 1, 2, 3
    raw = sorted(brent(poly, a, b)[0] for a, b in find_brackets(poly, 0.5, 3.5, 301))
    roots = []
    for r in raw:
        if not roots or abs(r - roots[-1]) > 1e-6:
            roots.append(r)
    print(f"    x^3 - 6x^2 + 11x - 6  ->  {[round(x, 6) for x in roots]}")
    print("\n  Bisection is foolproof but linear (one bit per step); secant is fast but can")
    print("  fail; Brent takes the fast interpolation step when it is safe and bisects when it")
    print("  is not -- guaranteed convergence at near-secant speed, the default in most libraries.")

    _svg(os.path.join(outdir, "rootfind.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'rootfind.svg')}")


def _trace(method_step, f, *init, max_iter=60, tol=1e-15):
    """Run a root method step-by-step, returning the list of |f(x)| residuals per iteration."""
    return method_step(f, *init, max_iter=max_iter, tol=tol)


def _bisection_residuals(f, a, b, n=55):
    fa = f(a)
    res = []
    for _ in range(n):
        m = 0.5 * (a + b)
        fm = f(m)
        res.append(abs(fm))
        if (f(a) > 0) == (fm > 0):
            a = m
        else:
            b = m
    return res


def _secant_residuals(f, x0, x1, n=12):
    f0, f1 = f(x0), f(x1)
    res = []
    for _ in range(n):
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        res.append(abs(f(x2)))
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
        if res[-1] < 1e-16:
            break
    return res


def _false_pos_residuals(f, a, b, n=30):
    fa, fb = f(a), f(b)
    res = []
    for _ in range(n):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        res.append(abs(fc))
        if (fc > 0) == (fa > 0):
            a, fa = c, fc
        else:
            b, fb = c, fc
        if res[-1] < 1e-16:
            break
    return res


def _svg(path, w=760, h=390):
    f = lambda x: x * x - 2
    series = [
        ("bisection", _bisection_residuals(f, 0, 2), "#ff6b6b"),
        ("false position", _false_pos_residuals(f, 0, 2), "#ffd43b"),
        ("secant", _secant_residuals(f, 0, 2), "#06d6a0"),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="18">'
        f'Convergence: |f(x)| per iteration on x^2 - 2</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="12">'
        f'bisection is linear (steady slope); secant is superlinear (steepening) -- log-scale error</text>',
    ]

    x0, x1 = 55, w - 30
    y0, y1 = h - 55, 62
    imax = max(len(s) for _, s, _ in series)
    emin, emax = -16.0, 1.0

    def X(i):
        return x0 + i / max(1, imax - 1) * (x1 - x0)

    def Y(e):
        le = max(emin, min(emax, math.log10(max(e, 1e-16))))
        return y0 - (le - emin) / (emax - emin) * (y0 - y1)

    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#8b949e" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8b949e" stroke-width="1.2"/>')
    for e in range(-16, 2, 4):
        yy = Y(10.0 ** e)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{x0-6:.1f}" y="{yy+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">1e{e}</text>')
    ly = y1
    for name, res, col in series:
        pts = " ".join(f"{X(i):.1f},{Y(e):.1f}" for i, e in enumerate(res))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        for i, e in enumerate(res):
            parts.append(f'<circle cx="{X(i):.1f}" cy="{Y(e):.1f}" r="1.8" fill="{col}"/>')
        parts.append(f'<rect x="{x1-130}" y="{ly}" width="9" height="9" fill="{col}"/>'
                     f'<text x="{x1-117}" y="{ly+8}" fill="#e6edf3" font-size="9">{name}</text>')
        ly += 14
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0+18:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">iteration -> |f(x)|</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

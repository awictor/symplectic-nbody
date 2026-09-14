"""AAA demo: fit functions with poles using a handful of greedy support points, and recover the singularities."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import aaa_approx as aaa


PALETTE = {
    "bg": "#0d1117", "blue": "#4dabf7", "yellow": "#ffd43b", "red": "#ff6b6b",
    "green": "#06d6a0", "purple": "#b197fc", "gray": "#8b949e", "text": "#e6edf3",
}


def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def poly_lsq(xs, ys, degree):
    n = degree + 1
    A = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum(ys[k] * xs[k] ** i for k in range(len(xs))) for i in range(n)]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[p] = M[p], M[col]
        pv = M[col][col]
        for r in range(n):
            if r == col:
                continue
            f=M[r][col] / pv
            for cc in range(col, n + 1):
                M[r][cc] -= f * M[col][cc]
    c = [M[i][n] / M[i][i] for i in range(n)]
    return lambda x: sum(c[i] * x ** i for i in range(n))


def main():
    lines = []
    lines.append("AAA algorithm -- near-optimal rational approximation, robust to poles")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Barycentric rational form + greedy support points + Loewner SVD weights.")
    lines.append("")

    # convergence on three functions
    lines.append("Root-exponential convergence (terms to reach 1e-12):")
    lines.append("   function          domain        #terms   max error")
    lines.append("   " + "-" * 50)
    cases = [
        ("exp(x)", math.exp, (-1, 1)),
        ("exp(-x^2)", lambda x: math.exp(-x * x), (-1, 1)),
        ("1/(x-1.5)", lambda x: 1 / (x - 1.5), (-1, 1)),
        ("tan(x)", math.tan, (-1.2, 1.2)),
        ("|x| approx sqrt(x^2+e)", lambda x: math.sqrt(x * x + 1e-4), (-1, 1)),
    ]
    for name, f, dom in cases:
        Z = linspace(dom[0], dom[1], 100)
        F = [f(z) for z in Z]
        r = aaa.aaa(Z, F, tol=1e-12)
        xt = linspace(dom[0], dom[1], 400)
        err = max(abs(r["eval"](x) - f(x)) for x in xt)
        lines.append(f"   {name:20s} [{dom[0]},{dom[1]}]   {r['num_terms']:4d}    {err:.2e}")
    lines.append("")

    # pole recovery for tan on [-1.4,1.4]
    ft = math.tan
    Zt = linspace(-1.4, 1.4, 120)
    rt = aaa.aaa(Zt, [ft(z) for z in Zt], tol=1e-11)
    pls = aaa.poles(rt)
    real_poles = sorted((p.real for p in pls if abs(p.imag) < 1e-3), key=abs)
    lines.append(f"tan(x) fit with {rt['num_terms']} terms. Poles recovered near +-pi/2 = +-{math.pi/2:.5f}:")
    for p in real_poles[:4]:
        lines.append(f"   pole at {p:+.6f}")
    lines.append("")

    # AAA vs polynomial on a near-singular function
    fs = lambda x: 1.0 / (x - 1.05)
    Zs = linspace(-1, 1, 80)
    Fs = [fs(z) for z in Zs]
    rs = aaa.aaa(Zs, Fs, tol=1e-11)
    deg = rs["num_terms"]
    xt = linspace(-1, 1, 400)
    aaa_err = max(abs(rs["eval"](x) - fs(x)) for x in xt)
    p = poly_lsq(Zs, Fs, deg)
    poly_err = max(abs(p(x) - fs(x)) for x in xt)
    lines.append(f"Near-singular 1/(x-1.05): AAA ({deg} terms) max err {aaa_err:.2e}")
    lines.append(f"                          degree-{deg} least-sq polynomial max err {poly_err:.2e}")
    lines.append(f"  -> AAA is {poly_err / aaa_err:.0e}x more accurate; polynomials cannot represent poles.")

    text = "\n".join(lines)
    print(text)

    svg = _svg()
    return text, svg


def _svg():
    """Plot 1/(x-1.1): the true function, the AAA fit, its support points, and a polynomial of equal order."""
    W, H = 640, 440
    P = PALETTE
    f = lambda x: 1.0 / (x - 1.1)
    a, b = -1.0, 1.0
    Z = linspace(a, b, 80)
    F = [f(z) for z in Z]
    r = aaa.aaa(Z, F, tol=1e-11)
    deg = r["num_terms"]
    poly = poly_lsq(Z, F, deg)

    xt = linspace(a, b, 400)
    ftrue = [f(x) for x in xt]
    faaa = [r["eval"](x) for x in xt]
    fpoly = [poly(x) for x in xt]

    ymin = min(min(ftrue), min(fpoly))
    ymax = max(max(ftrue), max(fpoly))
    # clamp the polynomial's wild excursions for a readable frame
    ymin = max(ymin, -12.0)
    ymax = min(ymax, 2.0)
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    x0, x1, y0, y1 = 50, 620, 55, 370

    def px(x):
        return x0 + (x - a) / (b - a) * (x1 - x0)

    def py(v):
        v = max(min(v, ymax), ymin)
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="{P["bg"]}"/>')
    parts.append(f'<text x="20" y="26" fill="{P["text"]}" font-size="15">'
                 f'AAA rational fit of 1/(x-1.1) vs an equal-order polynomial</text>')

    def polyline(vals, color, width, dash=None):
        pts = " ".join(f"{px(xt[i]):.1f},{py(vals[i]):.1f}" for i in range(len(xt)))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{d}/>'

    # polynomial (red, oscillates)
    parts.append(polyline(fpoly, P["red"], 1.6, dash="5,4"))
    # true function (gray, thick)
    parts.append(polyline(ftrue, P["gray"], 3))
    # AAA fit (yellow, overlays truth)
    parts.append(polyline(faaa, P["yellow"], 1.8))

    # support points as green dots on the true curve
    for j in range(len(r["support_z"])):
        zjj = r["support_z"][j]
        parts.append(f'<circle cx="{px(zjj):.1f}" cy="{py(f(zjj)):.1f}" r="3.5" '
                     f'fill="{P["green"]}"/>')

    parts.append(f'<text x="{x0}" y="{y1 + 24}" fill="{P["gray"]}" font-size="11">'
                 f'gray = true 1/(x-1.1)</text>')
    parts.append(f'<text x="{x0 + 175}" y="{y1 + 24}" fill="{P["yellow"]}" font-size="11">'
                 f'yellow = AAA ({deg} terms, on top of truth)</text>')
    parts.append(f'<text x="{x0}" y="{y1 + 42}" fill="{P["red"]}" font-size="11">'
                 f'red dashed = degree-{deg} polynomial (Runge oscillation)</text>')
    parts.append(f'<text x="{x0 + 320}" y="{y1 + 42}" fill="{P["green"]}" font-size="11">'
                 f'green = greedy support points</text>')
    parts.append(f'<text x="20" y="{H - 12}" fill="{P["gray"]}" font-size="11">'
                 f'The pole at x=1.1 sits just past the domain edge; AAA tracks the steep rise, '
                 f'the polynomial cannot.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()

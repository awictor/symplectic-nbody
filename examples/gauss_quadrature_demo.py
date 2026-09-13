"""Demo: Gaussian quadrature on infinite domains -- Gauss-Hermite and Gauss-Laguerre.

Computes integrals over the whole line and the half-line that ordinary quadrature cannot touch, shows
exactness for polynomials and geometric convergence for smooth functions, evaluates a Gaussian
expectation, and draws the node placement and the convergence curve.

    python examples/gauss_quadrature_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gauss_quadrature import (gauss_hermite, gauss_laguerre, integrate_hermite,  # noqa: E402
                              integrate_laguerre, gaussian_expectation)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Gaussian quadrature on infinite domains: exact with a handful of points\n")

    print("  Gauss-Hermite (weight e^{-x^2} over the whole line):")
    print(f"    integral e^{{-x^2}} dx        = {integrate_hermite(lambda x: 1.0, 3):.10f}  "
          f"(exact sqrt(pi) = {math.sqrt(math.pi):.10f})")
    print(f"    integral cos(x) e^{{-x^2}} dx  = {integrate_hermite(math.cos, 8):.10f}  "
          f"(exact = {math.sqrt(math.pi)*math.exp(-0.25):.10f})")
    print(f"    5-point nodes: {[round(x,4) for x in sorted(gauss_hermite(5)[0])]}  (symmetric)\n")

    print("  Gauss-Laguerre (weight e^{-x} over the half-line):")
    print(f"    integral x^3 e^{{-x}} dx = {integrate_laguerre(lambda x: x**3, 5):.6f}  (exact 3! = 6)")
    for s in (0.5, 2.5, 4.5):
        got = integrate_laguerre(lambda x, s=s: x ** s, 30)
        print(f"    Gamma({s+1}) = integral x^{s} e^{{-x}} = {got:.6f}  (exact {math.gamma(s+1):.6f})")
    print()

    print("  Gaussian expectations via change of variables x = mu + sqrt(2) sigma t:")
    print(f"    E[x^2] for N(0,1)       = {gaussian_expectation(lambda x: x*x, 0, 1, 5):.6f}  (=1)")
    print(f"    E[e^x]  for N(0,1)      = {gaussian_expectation(math.exp, 0, 1, 20):.6f}  "
          f"(= sqrt(e) = {math.sqrt(math.e):.6f})")
    print(f"    P-ish: E[max(x,0)] N(0,1) = {gaussian_expectation(lambda x: max(x,0), 0, 1, 40):.6f}  "
          f"(= 1/sqrt(2pi) = {1/math.sqrt(2*math.pi):.6f})\n")

    print("  The n nodes are the roots of the degree-n orthogonal polynomial for the weight, found as")
    print("  the eigenvalues of a tridiagonal Jacobi matrix (Golub-Welsch); the rule is then exact for")
    print("  every polynomial up to degree 2n-1 and converges geometrically on smooth integrands.")

    _svg(os.path.join(outdir, "gauss_quadrature.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'gauss_quadrature.svg')}")


def _svg(path, width=760, height=430):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="16">'
        f'Gauss-Hermite nodes under e^{{-x^2}} (top); convergence on cos(x) (bottom)</text>',
    ]

    # top: nodes + weights of a 9-point Hermite rule over the weight function
    ox, oy = 60, 180
    pw, ph = width - 100, 120
    nodes, weights = gauss_hermite(9)
    xr = 4.0
    wmax = max(weights)

    def px(x):
        return ox + (x + xr) / (2 * xr) * pw

    # weight curve e^{-x^2}
    curve = []
    steps = 200
    for i in range(steps + 1):
        x = -xr + 2 * xr * i / steps
        curve.append(f"{px(x):.1f},{oy - math.exp(-x*x)*ph:.1f}")
    parts.append(f'<polyline points="{" ".join(curve)}" fill="none" stroke="#4dabf7" '
                 f'stroke-width="1.5"/>')
    parts.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+pw}" y2="{oy}" stroke="#30363d" stroke-width="1"/>')
    # node stems, height ~ weight
    for x, w in zip(nodes, weights):
        h = w / wmax * ph * 0.9
        parts.append(f'<line x1="{px(x):.1f}" y1="{oy}" x2="{px(x):.1f}" y2="{oy-h:.1f}" '
                     f'stroke="#ffd43b" stroke-width="2"/>')
        parts.append(f'<circle cx="{px(x):.1f}" cy="{oy-h:.1f}" r="3.5" fill="#ffd43b"/>')
    parts.append(f'<text x="{ox}" y="{oy+18}" fill="#8b949e" font-size="11">'
                 f'blue: weight e^{{-x^2}}   yellow stems: 9 nodes, height = quadrature weight</text>')

    # bottom: convergence of integral cos(x) e^{-x^2}
    bx, by = 60, 400
    bw, bh = width - 100, 150
    exact = math.sqrt(math.pi) * math.exp(-0.25)
    ns = list(range(2, 13))
    errs = [max(abs(integrate_hermite(math.cos, n) - exact), 1e-17) for n in ns]
    logs = [math.log10(e) for e in errs]
    lo, hi = -17, 0

    def cx(n):
        return bx + (n - ns[0]) / (ns[-1] - ns[0]) * bw

    def cy(le):
        return by - (le - lo) / (hi - lo) * bh

    for le, lab in ((0, "1"), (-8, "1e-8"), (-16, "1e-16")):
        y = cy(le)
        parts.append(f'<line x1="{bx}" y1="{y:.1f}" x2="{bx+bw}" y2="{y:.1f}" stroke="#21262d" '
                     f'stroke-width="0.6"/>')
        parts.append(f'<text x="{bx-6}" y="{y+4:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{lab}</text>')
    pts = " ".join(f"{cx(n):.1f},{cy(le):.1f}" for n, le in zip(ns, logs))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#06d6a0" stroke-width="2"/>')
    for n, le in zip(ns, logs):
        parts.append(f'<circle cx="{cx(n):.1f}" cy="{cy(le):.1f}" r="3" fill="#06d6a0"/>')
    parts.append(f'<text x="{bx+bw/2:.0f}" y="{by+22}" fill="#8b949e" font-size="11" '
                 f'text-anchor="middle">number of nodes n -- error plunges geometrically</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

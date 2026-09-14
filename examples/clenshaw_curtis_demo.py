"""Demo: Clenshaw-Curtis quadrature -- spectral accuracy from Chebyshev nodes vs the trapezoidal rule.

Integrates a smooth function with both Clenshaw-Curtis and the trapezoidal rule at matched node
counts, showing Clenshaw-Curtis converging spectrally (error dropping like a cliff) while trapezoid
crawls at O(1/n^2). Draws the error-vs-n convergence on a log scale and the clustered Chebyshev nodes.

    python examples/clenshaw_curtis_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clenshaw_curtis import integrate, trapezoid, nodes_weights  # noqa: E402


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Clenshaw-Curtis quadrature: spectral accuracy from Chebyshev-clustered nodes\n")

    # a smooth, non-periodic integrand
    f = lambda x: 1.0 / (2.0 + math.cos(3 * x)) * math.exp(-x)
    a, b = 0.0, 2.0
    ref = integrate(f, a, b, 512)
    print(f"  integrand: e^-x / (2 + cos 3x) on [{a}, {b}]")
    print(f"  reference integral (n=512): {ref:.14f}\n")

    print(f"    {'n':>5}{'Clenshaw-Curtis err':>22}{'trapezoid err':>18}")
    cc_errs, tr_errs = [], []
    ns = [4, 8, 16, 32, 64]
    for n in ns:
        cc = abs(integrate(f, a, b, n) - ref)
        tr = abs(trapezoid(f, a, b, n) - ref)
        cc_errs.append(cc)
        tr_errs.append(tr)
        print(f"    {n:>5}{cc:>22.2e}{tr:>18.2e}")

    print(f"\n  Clenshaw-Curtis error falls off a cliff (spectral) while trapezoid decays like")
    print(f"  O(1/n^2). At n=32 Clenshaw-Curtis is already near machine precision.")
    print(f"  Its nodes cluster toward the endpoints (Chebyshev extrema), killing edge error,")
    print(f"  and they nest -- doubling n reuses every previous sample.")

    _svg(os.path.join(outdir, "clenshaw_curtis.svg"), ns, cc_errs, tr_errs, a, b)
    print(f"\n  wrote {os.path.join(outdir, 'clenshaw_curtis.svg')}")


def _svg(path, ns, cc, tr, a, b, width=760, height=420):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Quadrature error vs node count: Clenshaw-Curtis (spectral) vs trapezoid (O(1/n^2))</text>',
    ]
    ox, oy, ow, oh = 60, 55, width - 110, height - 160
    allv = [max(e, 1e-16) for e in cc + tr]
    lo = math.log10(min(allv))
    hi = math.log10(max(allv))

    def px(i):
        return ox + ow * i / (len(ns) - 1)

    def py(e):
        lg = math.log10(max(e, 1e-16))
        return oy + oh * (1 - (lg - lo) / (hi - lo + 1e-12))

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    d = math.floor(lo)
    while d <= hi:
        y = py(10 ** d)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+ow}" y2="{y:.1f}" stroke="#161b22"/>')
        parts.append(f'<text x="{ox-6}" y="{y+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">1e{int(d)}</text>')
        d += 2

    def poly(errs, color):
        pts = " ".join(f"{px(i):.1f},{py(errs[i]):.1f}" for i in range(len(ns)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i in range(len(ns)):
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(errs[i]):.1f}" r="3" fill="{color}"/>')

    poly(cc, "#4dabf7")
    poly(tr, "#ff922b")
    for i, n in enumerate(ns):
        parts.append(f'<text x="{px(i):.1f}" y="{oy+oh+14:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">n={n}</text>')
    parts.append(f'<rect x="{ox+ow-160}" y="{oy+6}" width="12" height="3" fill="#4dabf7"/>')
    parts.append(f'<text x="{ox+ow-144}" y="{oy+10}" fill="#e6edf3" font-size="10">Clenshaw-Curtis</text>')
    parts.append(f'<rect x="{ox+ow-160}" y="{oy+22}" width="12" height="3" fill="#ff922b"/>')
    parts.append(f'<text x="{ox+ow-144}" y="{oy+26}" fill="#e6edf3" font-size="10">trapezoid</text>')

    # Chebyshev node clustering below
    xs, _ = nodes_weights(16, a, b)
    ny = oy + oh + 55
    parts.append(f'<text x="{ox}" y="{ny-10:.0f}" fill="#8b949e" font-size="10">'
                 f'Clenshaw-Curtis nodes cluster toward the endpoints:</text>')
    for x in xs:
        sxv = ox + ow * (x - a) / (b - a)
        parts.append(f'<circle cx="{sxv:.1f}" cy="{ny:.1f}" r="3" fill="#06d6a0"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

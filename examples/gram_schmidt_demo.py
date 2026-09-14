"""Demo: Gram-Schmidt orthogonalization -- why modified beats classical, and orthogonal polynomials.

Orthogonalizes an ill-conditioned (Hilbert) basis with both classical and modified Gram-Schmidt,
showing the classical version's catastrophic loss of orthogonality against the modified version's near
machine precision, and generates the Legendre polynomials by Gram-Schmidt on the monomials. Draws the
orthogonality error vs basis size for both.

    python examples/gram_schmidt_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gram_schmidt import (  # noqa: E402
    classical_gram_schmidt, modified_gram_schmidt, orthogonality_error, orthogonal_polynomials,
)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Gram-Schmidt: classical vs modified, and the order of operations that decides stability\n")

    print(f"  orthogonality error ||Q^T Q - I|| on the ill-conditioned Hilbert basis:")
    print(f"    {'size':>6}{'classical (CGS)':>18}{'modified (MGS)':>18}")
    cgs_errs, mgs_errs, sizes = [], [], []
    for n in [4, 6, 8, 10, 12]:
        hilbert = [[1.0 / (i + j + 1) for i in range(n)] for j in range(n)]
        Qc, _ = classical_gram_schmidt(hilbert)
        Qm, _ = modified_gram_schmidt(hilbert)
        ce = orthogonality_error(Qc)
        me = orthogonality_error(Qm)
        cgs_errs.append(ce)
        mgs_errs.append(me)
        sizes.append(n)
        print(f"    {n:>6}{ce:>18.2e}{me:>18.2e}")
    print(f"  Classical Gram-Schmidt collapses (error near 1 -- the vectors are not orthogonal at all)")
    print(f"  while modified stays near machine precision. Same formula, different operation order.\n")

    # orthogonal polynomials
    print(f"  Gram-Schmidt on the monomials 1, x, x^2, ... under the L2 inner product on [-1,1]")
    print(f"  produces the (normalized) Legendre polynomials:")
    op = orthogonal_polynomials(4)
    names = ["P0", "P1", "P2", "P3", "P4"]
    for k, c in enumerate(op):
        terms = " + ".join(f"{coef:+.3f} x^{p}" for p, coef in enumerate(c) if abs(coef) > 1e-6)
        print(f"    {names[k]}: {terms}")

    print(f"\n  These orthonormal polynomials are exactly what Gaussian quadrature and Chebyshev/")
    print(f"  Legendre spectral methods are built on -- Gram-Schmidt manufactures them from scratch.")

    _svg(os.path.join(outdir, "gram_schmidt.svg"), sizes, cgs_errs, mgs_errs)
    print(f"\n  wrote {os.path.join(outdir, 'gram_schmidt.svg')}")


def _svg(path, sizes, cgs, mgs, width=760, height=380):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Loss of orthogonality vs basis size: classical Gram-Schmidt collapses, modified holds</text>',
    ]
    ox, oy, ow, oh = 60, 55, width - 100, height - 110
    allv = [max(e, 1e-18) for e in cgs + mgs]
    lo = math.log10(min(allv))
    hi = math.log10(max(allv))

    def px(i):
        return ox + ow * i / (len(sizes) - 1)

    def py(e):
        return oy + oh * (1 - (math.log10(max(e, 1e-18)) - lo) / (hi - lo + 1e-12))

    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="none" stroke="#30363d"/>')
    d = math.floor(lo)
    while d <= hi:
        y = py(10 ** d)
        parts.append(f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+ow}" y2="{y:.1f}" stroke="#161b22"/>')
        parts.append(f'<text x="{ox-6}" y="{y+3:.1f}" fill="#8b949e" font-size="8" '
                     f'text-anchor="end">1e{int(d)}</text>')
        d += 3

    def poly(errs, color):
        pts = " ".join(f"{px(i):.1f},{py(errs[i]):.1f}" for i in range(len(sizes)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i in range(len(sizes)):
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(errs[i]):.1f}" r="3" fill="{color}"/>')

    poly(cgs, "#ff6b6b")
    poly(mgs, "#06d6a0")
    for i, n in enumerate(sizes):
        parts.append(f'<text x="{px(i):.1f}" y="{oy+oh+14:.0f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">n={n}</text>')
    parts.append(f'<rect x="{ox+ow-150}" y="{oy+6}" width="12" height="3" fill="#ff6b6b"/>')
    parts.append(f'<text x="{ox+ow-134}" y="{oy+10}" fill="#e6edf3" font-size="10">classical</text>')
    parts.append(f'<rect x="{ox+ow-150}" y="{oy+22}" width="12" height="3" fill="#06d6a0"/>')
    parts.append(f'<text x="{ox+ow-134}" y="{oy+26}" fill="#e6edf3" font-size="10">modified</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

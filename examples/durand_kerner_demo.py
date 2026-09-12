"""Demo: Durand-Kerner -- all roots of a polynomial at once, in the complex plane.

Finds every root (real and complex) of several polynomials, checks them by reconstruction, and draws
the roots as points in the complex plane with the unit circle for reference.

    python examples/durand_kerner_demo.py [output_dir]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from durand_kerner import roots, from_roots, match_roots, residual, real_roots  # noqa: E402


def fmt(z, p=4):
    if abs(z.imag) < 1e-7:
        return f"{z.real:.{p}f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.{p}f} {sign} {abs(z.imag):.{p}f}i"


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Durand-Kerner: all n roots of a degree-n polynomial simultaneously\n")

    polys = [
        ("x^2 - 3x + 2", [1, -3, 2]),
        ("x^3 - 6x^2 + 11x - 6", [1, -6, 11, -6]),
        ("x^2 + 1", [1, 0, 1]),
        ("x^4 - 1", [1, 0, 0, 0, -1]),
        ("x^3 + 1", [1, 0, 0, 1]),
    ]
    for name, c in polys:
        rs = roots(c)
        maxres = max(residual(c, r) for r in rs)
        print(f"  {name:22s} roots: {', '.join(fmt(z) for z in sorted(rs, key=lambda z:(z.real, z.imag)))}")
        print(f"  {'':22s} max |p(root)| = {maxres:.1e}")

    # a polynomial built from chosen complex roots, recovered
    known = [2 + 3j, 2 - 3j, -1 + 0j, 0.5 + 0j]
    coeffs = from_roots(known)
    found = roots(coeffs)
    print(f"\n  built from roots {[fmt(z) for z in known]}:")
    print(f"    recovered: {[fmt(z) for z in found]}")
    print(f"    match: {match_roots(found, known)}")

    print("\n  Each guess updates by r_i <- r_i - p(r_i) / prod_{j!=i}(r_i - r_j), the Weierstrass")
    print("  correction: it divides out the influence of the other roots. From spread complex seeds")
    print("  all n roots converge together, quadratically -- no bracketing, no derivative, no")
    print("  deflation, and it finds complex roots the real bracketing methods simply cannot see.")

    # roots of x^8 - 1: the eighth roots of unity, on the unit circle
    unity = roots([1] + [0] * 7 + [-1])
    _svg(os.path.join(outdir, "durand_kerner.svg"), unity)
    print(f"\n  wrote {os.path.join(outdir, 'durand_kerner.svg')} (the 8th roots of unity)")


def _svg(path, rts, width=440, height=440):
    cx, cy = width / 2, height / 2
    scale = 150

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        f'<text x="20" y="26" fill="#e6edf3" font-size="15">'
        f'Roots of x^8 - 1 in the complex plane</text>',
        f'<text x="20" y="44" fill="#8b949e" font-size="11">'
        f'the eight roots of unity, evenly spaced on the unit circle</text>',
    ]
    # axes and unit circle
    parts.append(f'<line x1="20" y1="{cy}" x2="{width-20}" y2="{cy}" stroke="#30363d"/>')
    parts.append(f'<line x1="{cx}" y1="60" x2="{cx}" y2="{height-20}" stroke="#30363d"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{scale}" fill="none" stroke="#484f58" '
                 f'stroke-dasharray="4 4"/>')

    for z in rts:
        x = cx + z.real * scale
        y = cy - z.imag * scale
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#ff6b6b"/>')
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#4dabf7" '
                     f'stroke-width="0.8" opacity="0.5"/>')

    parts.append(f'<text x="{cx+scale+4:.0f}" y="{cy-4:.0f}" fill="#8b949e" font-size="10">Re</text>')
    parts.append(f'<text x="{cx+4:.0f}" y="72" fill="#8b949e" font-size="10">Im</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: the London/Meissner effect -- a superconductor expelling magnetic field.

Prints the penetration depth for several carrier densities and the type I/II classification
of real superconductors, then draws the exponential field decay into the surface (the
Meissner screening) and where materials fall across the type-I/II boundary.

    python examples/london_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from london import (penetration_depth, field_profile, ginzburg_landau_parameter,  # noqa: E402
                    is_type_ii, vortex_flux, KAPPA_C)


# (name, lambda_L nm, xi nm)
MATERIALS = [("aluminium", 16, 1600), ("tin", 34, 230), ("niobium", 40, 38),
             ("Nb-Ti", 300, 4), ("YBCO (high-Tc)", 150, 1.5)]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("London/Meissner: B(x) = B0 exp(-x/lambda_L), lambda_L = sqrt(m/(mu0 n_s q^2))\n")
    print("  Penetration depth vs superconducting carrier density:")
    for n in (1e28, 4e28, 1e29):
        print(f"    n_s = {n:.0e} /m^3  ->  lambda_L = {penetration_depth(n)*1e9:.1f} nm")

    print("\n  Type classification (kappa = lambda_L/xi, boundary 1/sqrt2 = %.3f):" % KAPPA_C)
    print(f"  {'material':<18}{'lambda (nm)':>12}{'xi (nm)':>9}{'kappa':>9}{'type':>9}")
    for name, lam, xi in MATERIALS:
        k = ginzburg_landau_parameter(lam, xi)
        print(f"  {name:<18}{lam:>12}{xi:>9}{k:>9.2f}{'II' if is_type_ii(k) else 'I':>9}")

    print("\n  Type I expels field until it abruptly goes normal; type II lets field thread")
    print("  through as quantized vortices (each carrying %.2e Wb) between H_c1 and H_c2 --" % vortex_flux())
    print("  which is how Nb-Ti and high-Tc magnets survive the huge fields of MRI and fusion.")

    _svg(os.path.join(outdir, "london.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'london.svg')}")


def _svg(path, size=720, pad=76):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Meissner effect</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'field decays into the superconductor over lambda_L (top); type I vs II by kappa (bottom)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.52

    # --- top: B(x) decay, with the vacuum region and the superconductor ---
    ty0, ty1 = mid - 26, pad + 44
    lam_px = (x1 - x0) * 0.16     # one penetration depth on screen
    surf = x0 + (x1 - x0) * 0.35  # surface position
    def BY(B):
        return ty0 - B * (ty0 - ty1)
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    # vacuum region (constant field)
    parts.append(f'<rect x="{x0:.1f}" y="{ty1:.1f}" width="{surf-x0:.1f}" height="{ty0-ty1:.1f}" '
                 f'fill="#4dabf7" opacity="0.06"/>')
    parts.append(f'<line x1="{x0:.1f}" y1="{BY(1.0):.1f}" x2="{surf:.1f}" y2="{BY(1.0):.1f}" '
                 f'stroke="#4dabf7" stroke-width="2.4"/>')
    # superconductor region (exponential decay)
    parts.append(f'<rect x="{surf:.1f}" y="{ty1:.1f}" width="{x1-surf:.1f}" height="{ty0-ty1:.1f}" '
                 f'fill="#30363d" opacity="0.4"/>')
    pts = []
    n = 120
    for i in range(n + 1):
        xx = surf + (x1 - surf) * i / n
        x_depth = (xx - surf) / lam_px
        pts.append(f"{xx:.1f},{BY(math.exp(-x_depth)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.6"/>')
    # mark one penetration depth
    parts.append(f'<line x1="{surf+lam_px:.1f}" y1="{ty0:.1f}" x2="{surf+lam_px:.1f}" y2="{BY(math.exp(-1)):.1f}" '
                 f'stroke="#ffd43b" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<circle cx="{surf+lam_px:.1f}" cy="{BY(math.exp(-1)):.1f}" r="4" fill="#ffd43b"/>')
    parts.append(f'<text x="{surf+lam_px+6:.1f}" y="{BY(math.exp(-1)):.1f}" fill="#ffd43b" font-size="10">'
                 f'lambda_L (B -> B0/e)</text>')
    parts.append(f'<line x1="{surf:.1f}" y1="{ty1:.1f}" x2="{surf:.1f}" y2="{ty0:.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{(x0+surf)/2:.1f}" y="{ty1+14:.1f}" fill="#4dabf7" font-size="10" '
                 f'text-anchor="middle">vacuum (field B0)</text>')
    parts.append(f'<text x="{(surf+x1)/2:.1f}" y="{ty1+14:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">superconductor (field expelled)</text>')

    # --- bottom: kappa scale with the type boundary ---
    by = size * 0.80
    lx0, lx1 = x0, x1
    kmin, kmax = -1.0, 2.0   # log10 kappa
    def KX(logk):
        return lx0 + (logk - kmin) / (kmax - kmin) * (lx1 - lx0)
    parts.append(f'<line x1="{lx0:.1f}" y1="{by:.1f}" x2="{lx1:.1f}" y2="{by:.1f}" '
                 f'stroke="#8b949e" stroke-width="1.5"/>')
    # boundary at kappa = 1/sqrt2
    kb = KX(math.log10(KAPPA_C))
    parts.append(f'<line x1="{kb:.1f}" y1="{by-40:.1f}" x2="{kb:.1f}" y2="{by+12:.1f}" '
                 f'stroke="#ff6b6b" stroke-width="1.6" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{kb:.1f}" y="{by-46:.1f}" fill="#ff6b6b" font-size="10" '
                 f'text-anchor="middle">kappa = 1/sqrt2</text>')
    parts.append(f'<text x="{KX(-0.5):.1f}" y="{by+28:.1f}" fill="#4dabf7" font-size="11" '
                 f'text-anchor="middle">type I (expels field)</text>')
    parts.append(f'<text x="{KX(1.2):.1f}" y="{by+28:.1f}" fill="#06d6a0" font-size="11" '
                 f'text-anchor="middle">type II (flux vortices)</text>')
    for name, lam, xi in MATERIALS:
        k = lam / xi
        kx = KX(math.log10(k))
        col = "#06d6a0" if k > KAPPA_C else "#4dabf7"
        parts.append(f'<circle cx="{kx:.1f}" cy="{by:.1f}" r="4.5" fill="{col}"/>')
        parts.append(f'<text x="{kx:.1f}" y="{by-8:.1f}" fill="{col}" font-size="9" '
                     f'text-anchor="middle">{name.split(" ")[0]}</text>')
    for e in (-1, 0, 1, 2):
        parts.append(f'<text x="{KX(e):.1f}" y="{by+44:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">kappa=10^{e}</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

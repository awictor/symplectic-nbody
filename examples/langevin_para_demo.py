"""Demo: Langevin paramagnetism -- moments aligning against thermal noise.

Prints the alignment fraction across field/temperature and the Curie-law susceptibility, then
draws the Langevin function L(x) rising from the x/3 Curie slope to full saturation, and the
1/T Curie susceptibility.

    python examples/langevin_para_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langevin_para import (langevin, reduced_field, saturation_fraction,  # noqa: E402
                           curie_susceptibility, curie_constant, BOHR_MAGNETON)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Langevin paramagnetism: m/mu = L(x) = coth(x) - 1/x, x = mu B / kT\n")
    mu = 5 * BOHR_MAGNETON   # a modest atomic moment
    print("  Alignment fraction L(x) for a 5-Bohr-magneton moment:")
    print(f"  {'B (T)':>8}{'T (K)':>8}{'x':>10}{'aligned':>10}")
    for B, T in ((1, 300), (10, 300), (10, 4), (50, 1)):
        x = reduced_field(mu, B, T)
        print(f"  {B:>8}{T:>8}{x:>10.3f}{saturation_fraction(mu, B, T)*100:>8.1f} %")

    print("\n  Curie law chi = C/T (C = n mu^2/3k):")
    for T in (1, 10, 100, 300):
        print(f"    T = {T:>4} K  ->  chi (per moment) = {curie_susceptibility(mu, T):.3e}")

    print("\n  Weak field or high temperature: L(x) ~ x/3, so susceptibility falls as 1/T --")
    print("  Curie's law, the fingerprint of a paramagnet. Strong field or low temperature:")
    print("  every moment aligns and L saturates at 1, the magnetization can grow no further.")

    _svg(os.path.join(outdir, "langevin_para.svg"), mu)
    print(f"\n  wrote {os.path.join(outdir, 'langevin_para.svg')}")


def _svg(path, mu, size=720, pad=76):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">Langevin paramagnetism</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'alignment L(x) from the x/3 Curie slope to saturation (top); chi ~ 1/T (bottom)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.54

    # --- top: L(x) ---
    ty0, ty1 = mid - 26, pad + 44
    x_max = 8.0
    def LX(x):
        return x0 + x / x_max * (x1 - x0)
    def LY(L):
        return ty0 - L * (ty0 - ty1)
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    for L in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{x0-6:.1f}" y="{LY(L)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{L:.1f}</text>')
    # saturation line
    parts.append(f'<line x1="{x0}" y1="{LY(1.0):.1f}" x2="{x1}" y2="{LY(1.0):.1f}" '
                 f'stroke="#ff6b6b" stroke-width="0.8" stroke-dasharray="4 4" opacity="0.6"/>')
    parts.append(f'<text x="{x1-4:.1f}" y="{LY(1.0)-4:.1f}" fill="#ff6b6b" font-size="10" '
                 f'text-anchor="end">saturation L=1</text>')
    # Langevin curve
    pts = []
    n = 200
    for i in range(n + 1):
        x = x_max * i / n
        pts.append(f"{LX(x):.1f},{LY(langevin(x)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.8"/>')
    # x/3 Curie tangent
    parts.append(f'<line x1="{LX(0):.1f}" y1="{LY(0):.1f}" x2="{LX(3.0):.1f}" y2="{LY(1.0):.1f}" '
                 f'stroke="#06d6a0" stroke-width="1.4" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{LX(1.6):.1f}" y="{LY(0.62):.1f}" fill="#06d6a0" font-size="10">'
                 f'Curie slope L ~ x/3</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">x = mu B / kT   (alignment fraction L(x))</text>')

    # --- bottom: chi vs T (1/T) ---
    by0, by1 = size - pad, mid + 40
    T_min, T_max = 5.0, 300.0
    C = curie_constant(mu)
    chi_max = C / T_min
    def TX(T):
        return x0 + (T - T_min) / (T_max - T_min) * (x1 - x0)
    def CY(chi):
        return by0 - chi / chi_max * (by0 - by1)
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x1}" y2="{by0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x0}" y2="{by1}" stroke="#8b949e" stroke-width="1.4"/>')
    cpts = []
    T = T_min
    while T <= T_max:
        cpts.append(f"{TX(T):.1f},{CY(C / T):.1f}")
        T += 2.0
    parts.append(f'<polyline points="{" ".join(cpts)}" fill="none" stroke="#ff922b" stroke-width="2.6"/>')
    for T in (50, 100, 200, 300):
        parts.append(f'<text x="{TX(T):.1f}" y="{by0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{T} K</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">temperature (susceptibility chi = C/T, Curie law)</text>')
    parts.append(f'<text x="{TX(120):.1f}" y="{CY(C/60):.1f}" fill="#ff922b" font-size="11">'
                 f'chi diverges as T -> 0</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

"""Demo: BCS superconductivity -- the gap that kills resistance.

Prints the gap, BCS ratio and pair-breaking frequency for real superconductors, the isotope
effect, and the T_c from coupling, then draws the temperature-dependent gap closing at T_c for
several materials plus T_c versus electron-phonon coupling.

    python examples/bcs_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bcs import (gap_from_tc, gap_ratio, gap_at_temperature, critical_temperature,  # noqa: E402
                 isotope_shifted_tc, pair_breaking_frequency)

MEV = 1.602176634e-22

MATERIALS = [("aluminium", 1.2, "#4dabf7"), ("niobium", 9.3, "#06d6a0"),
             ("lead", 7.2, "#ff922b")]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("BCS superconductivity: 2 Delta(0) / (k_B T_c) = 3.53 (universal)\n")
    print(f"  {'material':<12}{'T_c (K)':>9}{'gap (meV)':>12}{'pair-break':>14}")
    for name, tc, _ in MATERIALS:
        g = gap_from_tc(tc)
        print(f"  {name:<12}{tc:>9.1f}{g/MEV:>12.3f}{pair_breaking_frequency(g)/1e12:>11.2f} THz")

    print("\n  Isotope effect (T_c ~ M^-1/2): mercury-198 vs mercury-202 (Hg T_c=4.15 K):")
    print(f"    Hg-202 / Hg-198 mass ratio 1.020  ->  T_c = {isotope_shifted_tc(4.15, 202/198):.3f} K")
    print("    heavier isotope, lower T_c -- proof phonons do the pairing.")

    print("\n  T_c from electron-phonon coupling (Debye freq 3e13 rad/s):")
    for lam in (0.2, 0.3, 0.4, 0.5):
        print(f"    lambda = {lam:.1f}  ->  T_c = {critical_temperature(3e13, lam):.2f} K")

    print("\n  Below T_c an energy gap opens at the Fermi surface: it costs 2 Delta to break a")
    print("  Cooper pair, so nothing scatters the condensate and resistance vanishes. The gap")
    print("  closes as sqrt(1 - T/Tc) toward T_c, where superconductivity ends.")

    _svg(os.path.join(outdir, "bcs.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'bcs.svg')}")


def _svg(path, size=720, pad=76):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">BCS superconductivity</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'the energy gap closes as sqrt(1-T/Tc) (top); T_c rises with coupling (bottom)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.54

    # --- top: gap vs T/Tc for each material (normalized) ---
    ty0, ty1 = mid - 26, pad + 44
    def GX(frac):
        return x0 + frac * (x1 - x0)
    def GY(g):
        return ty0 - g * (ty0 - ty1)      # g in units of Delta(0)
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x0}" y2="{ty1}" stroke="#8b949e" stroke-width="1.4"/>')
    for g in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{x0-6:.1f}" y="{GY(g)+3:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="end">{g:.1f}</text>')
    # single universal curve (normalized) since all share the sqrt shape
    n = 200
    pts = []
    for i in range(n + 1):
        frac = i / n
        pts.append(f"{GX(frac):.1f},{GY(gap_at_temperature(1.0, frac, 1.0)):.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4dabf7" stroke-width="2.8"/>')
    parts.append(f'<text x="{GX(0.15):.1f}" y="{GY(0.45):.1f}" fill="#4dabf7" font-size="11">'
                 f'Delta(T)/Delta(0) = sqrt(1 - T/Tc)</text>')
    parts.append(f'<circle cx="{GX(1.0):.1f}" cy="{GY(0.0):.1f}" r="4" fill="#ff6b6b"/>')
    parts.append(f'<text x="{GX(1.0)-4:.1f}" y="{GY(0.0)-8:.1f}" fill="#ff6b6b" font-size="10" '
                 f'text-anchor="end">gap closes at T_c</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+16:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">T / T_c   (gap, in units of Delta(0))</text>')

    # --- bottom: T_c vs coupling lambda ---
    by0, by1 = size - pad, mid + 40
    lam_min, lam_max = 0.1, 0.6
    tc_max = critical_temperature(3e13, lam_max)
    def LX(lam):
        return x0 + (lam - lam_min) / (lam_max - lam_min) * (x1 - x0)
    def TY(tc):
        return by0 - tc / tc_max * (by0 - by1)
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x1}" y2="{by0}" stroke="#8b949e" stroke-width="1.4"/>')
    parts.append(f'<line x1="{x0}" y1="{by0}" x2="{x0}" y2="{by1}" stroke="#8b949e" stroke-width="1.4"/>')
    cpts = []
    lam = lam_min
    while lam <= lam_max:
        cpts.append(f"{LX(lam):.1f},{TY(critical_temperature(3e13, lam)):.1f}")
        lam += 0.01
    parts.append(f'<polyline points="{" ".join(cpts)}" fill="none" stroke="#8338ec" stroke-width="2.6"/>')
    for lam in (0.2, 0.3, 0.4, 0.5):
        parts.append(f'<text x="{LX(lam):.1f}" y="{by0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{lam:.1f}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">electron-phonon coupling lambda  (T_c = 1.13 hbar wD e^-1/lambda / kB)</text>')
    parts.append(f'<text x="{LX(0.2):.1f}" y="{TY(tc_max*0.6):.1f}" fill="#8338ec" font-size="10">'
                 f'exponentially sensitive to coupling</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()

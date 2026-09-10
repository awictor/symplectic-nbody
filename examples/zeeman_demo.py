"""Demo: the Zeeman effect -- a magnetic field splitting spectral lines.

Prints the normal-Zeeman splitting versus field, Lande g-factors for common levels, and the
field inferred from a solar measurement, then draws the normal triplet fanning out with field
and an anomalous-Zeeman sublevel ladder.

    python examples/zeeman_demo.py [output_dir]
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zeeman import (bohr_magneton, normal_zeeman_shift_hz,  # noqa: E402
                    normal_zeeman_shift_wavelength, lande_g, anomalous_shift_hz,
                    field_from_splitting)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    print("Zeeman effect: delta_E = g_J m_J mu_B B splits levels (mu_B = %.3e J/T)\n"
          % bohr_magneton())
    print(f"  {'field':>8}{'normal split':>16}{'at 500 nm':>16}")
    for B in (0.1, 0.3, 1.0, 3.0):
        print(f"  {B:>6.1f} T{normal_zeeman_shift_hz(B)/1e9:>13.2f} GHz"
              f"{normal_zeeman_shift_wavelength(B, 500e-9)*1e12:>13.2f} pm")

    print("\n  Lande g-factors (anomalous Zeeman -- uneven splitting):")
    for name, j, l, s in (("2S1/2 (ground)", 0.5, 0, 0.5),
                          ("2P1/2", 0.5, 1, 0.5),
                          ("2P3/2", 1.5, 1, 0.5),
                          ("3D3 (pure orbital-ish)", 3, 2, 1)):
        print(f"    {name:<24} g = {lande_g(j, l, s):.3f}")

    B_sun = field_from_splitting(4.2e9)
    print("\n  A sunspot line split by 4.2 GHz implies B = %.2f T -- how magnetograms map the" % B_sun)
    print("  Sun's magnetic field. The normal triplet (g=1) is the Lorentz-triplet classical")
    print("  physics got right; the uneven anomalous patterns forced the discovery of spin.")

    _svg(os.path.join(outdir, "zeeman.svg"))
    print(f"\n  wrote {os.path.join(outdir, 'zeeman.svg')}")


def _svg(path, size=720, pad=72):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="monospace">',
        f'<rect width="{size}" height="{size}" fill="#0d1117"/>',
        f'<text x="20" y="34" fill="#e6edf3" font-size="18">The Zeeman effect</text>',
        f'<text x="20" y="52" fill="#8b949e" font-size="12">'
        f'a magnetic field fans one spectral line into shifted components (normal triplet, top)</text>',
    ]

    x0, x1 = pad, size - pad
    mid = size * 0.54

    # --- top: normal triplet fanning out with field ---
    ty0, ty1 = mid - 30, pad + 44
    Bmax = 3.0
    fmax = normal_zeeman_shift_hz(Bmax) / 1e9 * 1.2   # GHz range for +/- components
    def BX(B):
        return x0 + B / Bmax * (x1 - x0)
    def FY(df):
        # df in GHz, 0 at centre
        cy = (ty0 + ty1) / 2
        return cy - df / fmax * (ty0 - ty1) / 2
    parts.append(f'<line x1="{x0}" y1="{ty0}" x2="{x1}" y2="{ty0}" stroke="#8b949e" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{(ty0+ty1)/2:.1f}" x2="{x1}" y2="{(ty0+ty1)/2:.1f}" '
                 f'stroke="#21262d" stroke-width="1"/>')
    # three components: 0 and +/- shift
    for sign, col, lbl in ((0, "#ffd43b", "pi (unshifted)"),
                           (+1, "#4dabf7", "sigma+"), (-1, "#ff6b6b", "sigma-")):
        pts = []
        for i in range(61):
            B = Bmax * i / 60
            df = sign * normal_zeeman_shift_hz(B) / 1e9
            pts.append(f"{BX(B):.1f},{FY(df):.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        df_end = sign * normal_zeeman_shift_hz(Bmax) / 1e9
        parts.append(f'<text x="{x1-4:.1f}" y="{FY(df_end)+ (4 if sign<=0 else -4):.1f}" '
                     f'fill="{col}" font-size="10" text-anchor="end">{lbl}</text>')
    for B in (0, 1, 2, 3):
        parts.append(f'<text x="{BX(B):.1f}" y="{ty0+14:.1f}" fill="#8b949e" font-size="9" '
                     f'text-anchor="middle">{B} T</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{ty0+28:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">magnetic field -> (component frequency shift, GHz)</text>')

    # --- bottom: anomalous sublevel ladder for a J=3/2 level (g=4/3) at fixed field ---
    by = size * 0.80
    lx = size * 0.5
    g = lande_g(1.5, 1, 0.5)
    parts.append(f'<text x="{x0:.1f}" y="{mid+28:.1f}" fill="#8b949e" font-size="12">'
                 f'Anomalous Zeeman: a J=3/2 level (g = {g:.2f}) splits into 4 evenly spaced m_J sublevels:</text>')
    step = 34
    for mj, off in ((1.5, -1.5), (0.5, -0.5), (-0.5, 0.5), (-1.5, 1.5)):
        yy = by + off * step * g
        parts.append(f'<line x1="{lx-140:.1f}" y1="{yy:.1f}" x2="{lx+140:.1f}" y2="{yy:.1f}" '
                     f'stroke="#06d6a0" stroke-width="2"/>')
        parts.append(f'<text x="{lx+150:.1f}" y="{yy+4:.1f}" fill="#8b949e" font-size="10">'
                     f'm_J = {mj:+.1f}</text>')
    # original (unsplit) level dashed
    parts.append(f'<line x1="{lx-140:.1f}" y1="{by:.1f}" x2="{lx+140:.1f}" y2="{by:.1f}" '
                 f'stroke="#8b949e" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>')
    parts.append(f'<text x="{lx-150:.1f}" y="{by+4:.1f}" fill="#8b949e" font-size="10" '
                 f'text-anchor="end">B = 0</text>')

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


if __name__ == "__main__":
    main()
